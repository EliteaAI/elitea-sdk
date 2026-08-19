import asyncio

from elitea_sdk.runtime import _parallel_hitl_registry as registry


def test_offer_requires_attached_supervisor_and_commit_wakes_exact_root():
    async def scenario():
        thread_id = 'registry-root'
        decision = {
            'decision_id': 'decision-1',
            'interrupt_id': 'interrupt-1',
            'action': 'approve',
        }
        registry.register(thread_id)
        assert registry.offer(thread_id, decision) is False

        supervisor_id = registry.attach(thread_id, asyncio.get_running_loop())
        assert registry.advertise(
            thread_id, supervisor_id, [decision['interrupt_id']],
        )
        assert registry.offer(thread_id, decision) is True
        waiter = asyncio.create_task(registry.wait(thread_id, supervisor_id))
        await asyncio.sleep(0)
        assert not waiter.done()
        assert registry.commit(thread_id, decision) is True
        assert await waiter == [decision]
        assert registry.drain(thread_id, supervisor_id) == []
        registry.unregister(thread_id)

    asyncio.run(scenario())


def test_duplicate_commit_is_idempotent():
    async def scenario():
        thread_id = 'registry-dedup-root'
        decision = {
            'decision_id': 'decision-1',
            'interrupt_id': 'interrupt-1',
        }
        registry.register(thread_id)
        supervisor_id = registry.attach(thread_id, asyncio.get_running_loop())
        assert registry.advertise(
            thread_id, supervisor_id, [decision['interrupt_id']],
        )
        assert registry.offer(thread_id, decision)
        assert not registry.offer(thread_id, {
            **decision, 'decision_id': 'decision-duplicate',
        })
        assert registry.commit(thread_id, decision)
        assert registry.commit(thread_id, decision)
        assert not registry.offer(thread_id, {
            **decision, 'decision_id': 'decision-after-commit',
        })
        assert registry.drain(thread_id, supervisor_id) == [decision]
        assert registry.drain(thread_id, supervisor_id) == []
        registry.unregister(thread_id)

    asyncio.run(scenario())


def test_concurrent_nested_supervisors_receive_only_owned_interrupts():
    async def scenario():
        thread_id = 'registry-wide-root'
        first = {
            'decision_id': 'decision-first',
            'interrupt_id': 'interrupt-first',
        }
        second = {
            'decision_id': 'decision-second',
            'interrupt_id': 'interrupt-second',
        }
        registry.register(thread_id)
        first_id = registry.attach(thread_id, asyncio.get_running_loop())
        second_id = registry.attach(thread_id, asyncio.get_running_loop())
        assert registry.advertise(
            thread_id, first_id, [first['interrupt_id']],
        )
        assert registry.advertise(
            thread_id, second_id, [second['interrupt_id']],
        )

        first_waiter = asyncio.create_task(registry.wait(thread_id, first_id))
        second_waiter = asyncio.create_task(registry.wait(thread_id, second_id))
        await asyncio.sleep(0)
        assert registry.offer(thread_id, second)
        assert registry.commit(thread_id, second)
        assert await second_waiter == [second]
        assert not first_waiter.done()

        assert registry.offer(thread_id, first)
        assert registry.commit(thread_id, first)
        assert await first_waiter == [first]
        registry.unregister(thread_id)

    asyncio.run(scenario())


def test_ambiguous_interrupt_owner_is_rejected():
    async def scenario():
        thread_id = 'registry-ambiguous-root'
        decision = {
            'decision_id': 'decision-ambiguous',
            'interrupt_id': 'interrupt-shared',
        }
        registry.register(thread_id)
        first_id = registry.attach(thread_id, asyncio.get_running_loop())
        second_id = registry.attach(thread_id, asyncio.get_running_loop())
        assert registry.advertise(
            thread_id, first_id, [decision['interrupt_id']],
        )
        assert registry.advertise(
            thread_id, second_id, [decision['interrupt_id']],
        )
        assert registry.offer(thread_id, decision) is False
        registry.unregister(thread_id)

    asyncio.run(scenario())


def test_one_supervisor_drains_multiple_committed_cards_in_order():
    async def scenario():
        thread_id = 'registry-multi-card-root'
        decisions = [
            {'decision_id': 'decision-a', 'interrupt_id': 'interrupt-a'},
            {'decision_id': 'decision-b', 'interrupt_id': 'interrupt-b'},
        ]
        registry.register(thread_id)
        supervisor_id = registry.attach(thread_id, asyncio.get_running_loop())
        assert registry.advertise(
            thread_id,
            supervisor_id,
            [item['interrupt_id'] for item in decisions],
        )
        for decision in decisions:
            assert registry.offer(thread_id, decision)
            assert registry.commit(thread_id, decision)
        assert registry.drain(thread_id, supervisor_id) == decisions
        registry.unregister(thread_id)

    asyncio.run(scenario())


def test_unregister_wakes_fully_paused_supervisor_for_durable_fallback():
    async def scenario():
        thread_id = 'registry-shutdown-root'
        registry.register(thread_id)
        supervisor_id = registry.attach(thread_id, asyncio.get_running_loop())
        assert registry.advertise(thread_id, supervisor_id, ['interrupt-a'])
        waiter = asyncio.create_task(registry.wait(thread_id, supervisor_id))
        await asyncio.sleep(0)
        assert not waiter.done()

        registry.unregister(thread_id)

        assert await asyncio.wait_for(waiter, timeout=1) == []

    asyncio.run(scenario())


def test_wait_clears_delayed_notification_after_commit_was_consumed():
    class WaitSettled(Exception):
        pass

    class ObservableEvent(asyncio.Event):
        calls = 0

        async def wait(self):
            self.calls += 1
            if self.calls == 2:
                assert not self.is_set()
                raise WaitSettled
            return await super().wait()

    async def scenario():
        thread_id = 'registry-delayed-wakeup-root'
        decision = {
            'decision_id': 'decision-delayed',
            'interrupt_id': 'interrupt-delayed',
        }
        registry.register(thread_id)
        try:
            supervisor_id = registry.attach(
                thread_id, asyncio.get_running_loop(),
            )
            supervisor = registry._mailboxes[thread_id].supervisors[
                supervisor_id
            ]
            supervisor.wakeup = ObservableEvent()
            assert registry.advertise(
                thread_id, supervisor_id, [decision['interrupt_id']],
            )
            assert registry.offer(thread_id, decision)
            assert registry.commit(thread_id, decision)

            # Consume synchronously before call_soon_threadsafe(wakeup.set)
            # executes, then let that now-stale callback run.
            assert registry.drain(thread_id, supervisor_id) == [decision]
            await asyncio.sleep(0)

            assert supervisor.wakeup.is_set()
            try:
                await registry.wait(thread_id, supervisor_id)
            except WaitSettled:
                pass
            else:
                raise AssertionError('wait did not re-enter the parked state')
        finally:
            registry.unregister(thread_id)

    asyncio.run(scenario())
