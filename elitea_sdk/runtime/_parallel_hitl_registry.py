"""In-process decision mailbox for supervised parallel HITL fan-outs.

The indexer worker owns transport and registers an active root thread.  Every
SDK supervisor beneath that root attaches its own asyncio loop only while a
parallel Application batch is running and advertises the exact interrupts it
owns. Pub/sub callbacks may arrive on another thread, so commits wake only the
owning supervisor through ``loop.call_soon_threadsafe``.

Decisions use a two-phase offer/commit contract.  An offer only proves that the
owning worker is alive; it can never resume a child.  Core emits the matching
commit only after it durably records live ownership, which prevents a late
pub/sub delivery from racing the aggregate checkpoint fallback.
"""

from __future__ import annotations

import asyncio
import threading
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Dict, List, Optional


MAX_PENDING_OFFERS = 256
MAX_PENDING_COMMITS = 256
MAX_SEEN_DECISIONS = 512


@dataclass
class _Supervisor:
    loop: asyncio.AbstractEventLoop
    wakeup: asyncio.Event
    interrupt_ids: set[str] = field(default_factory=set)
    commits: Dict[str, dict] = field(default_factory=dict)


@dataclass
class _Offer:
    decision: dict
    supervisor_id: str


@dataclass
class _Mailbox:
    active: bool = True
    supervisors: Dict[str, _Supervisor] = field(default_factory=dict)
    offers: Dict[str, _Offer] = field(default_factory=dict)
    seen_commits: set[str] = field(default_factory=set)
    seen_interrupts: set[str] = field(default_factory=set)


_lock = threading.Lock()
_mailboxes: Dict[str, _Mailbox] = {}


def register(thread_id: str) -> None:
    """Register a root turn before its graph starts executing."""
    if not thread_id:
        return
    with _lock:
        mailbox = _mailboxes.setdefault(thread_id, _Mailbox())
        mailbox.active = True


def unregister(thread_id: str) -> None:
    """Drop all transient decision state after the root turn terminates."""
    if not thread_id:
        return
    with _lock:
        mailbox = _mailboxes.pop(thread_id, None)
        supervisors = list(mailbox.supervisors.values()) if mailbox else []
    # A supervisor may be parked with every child paused.  Wake it after the
    # mailbox is removed so it can publish/return its durable aggregate during
    # graceful worker teardown instead of leaking an await forever.
    for supervisor in supervisors:
        supervisor.loop.call_soon_threadsafe(supervisor.wakeup.set)


def is_active(thread_id: str) -> bool:
    with _lock:
        mailbox = _mailboxes.get(thread_id)
        return bool(mailbox and mailbox.active)


def attach(thread_id: str, loop: asyncio.AbstractEventLoop) -> str:
    """Attach one scoped supervisor beneath an interactive root mailbox.

    A root can contain multiple concurrent nested fan-outs, each running its
    own asyncio loop in a worker thread.  The returned id keeps their wakeups
    isolated while ``interrupt_id`` remains the public routing identity.
    """
    if not thread_id:
        return ''
    supervisor_id = uuid4().hex
    with _lock:
        # Only the worker transport may create a live mailbox via register().
        # SDK-only/offline execution has no decision channel and must fall back
        # to the durable aggregate instead of parking forever on an unreachable
        # in-process supervisor.
        mailbox = _mailboxes.get(thread_id)
        if mailbox is None or not mailbox.active:
            return ''
        mailbox.supervisors[supervisor_id] = _Supervisor(
            loop=loop,
            wakeup=asyncio.Event(),
        )
    return supervisor_id


def detach(thread_id: str, supervisor_id: Optional[str] = None) -> None:
    """Detach a completed fan-out without ending the surrounding root turn."""
    if not thread_id:
        return
    with _lock:
        mailbox = _mailboxes.get(thread_id)
        if not mailbox:
            return
        removed = (
            set(mailbox.supervisors)
            if supervisor_id is None
            else {supervisor_id}
        )
        for current_id in removed:
            mailbox.supervisors.pop(current_id, None)
        mailbox.offers = {
            decision_id: offer
            for decision_id, offer in mailbox.offers.items()
            if offer.supervisor_id not in removed
        }


def advertise(
    thread_id: str,
    supervisor_id: str,
    interrupt_ids: List[str],
) -> bool:
    """Advertise exact interrupts currently owned by one supervisor."""
    values = {
        str(value) for value in (interrupt_ids or []) if str(value or '')
    }
    if not thread_id or not supervisor_id or not values:
        return False
    with _lock:
        mailbox = _mailboxes.get(thread_id)
        supervisor = (
            mailbox.supervisors.get(supervisor_id) if mailbox else None
        )
        if not mailbox or not mailbox.active or supervisor is None:
            return False
        supervisor.interrupt_ids.update(values)
        return True


def withdraw(
    thread_id: str,
    supervisor_id: str,
    interrupt_ids: List[str],
) -> None:
    """Stop accepting decisions while a paused child is being reconstructed."""
    values = {str(value) for value in (interrupt_ids or [])}
    with _lock:
        mailbox = _mailboxes.get(thread_id)
        supervisor = (
            mailbox.supervisors.get(supervisor_id) if mailbox else None
        )
        if supervisor is not None:
            supervisor.interrupt_ids.difference_update(values)


def offer(thread_id: str, decision: dict) -> bool:
    """Stage a decision offer and report whether this root can own it live."""
    decision_id = str((decision or {}).get("decision_id") or "")
    interrupt_id = str((decision or {}).get("interrupt_id") or "")
    if not thread_id or not decision_id or not interrupt_id:
        return False
    with _lock:
        mailbox = _mailboxes.get(thread_id)
        if not mailbox or not mailbox.active:
            return False
        if decision_id in mailbox.seen_commits:
            return True
        if interrupt_id in mailbox.seen_interrupts:
            return False
        if any(
            offer.decision.get('interrupt_id') == interrupt_id
            and offer.decision.get('decision_id') != decision_id
            for offer in mailbox.offers.values()
        ):
            return False
        owners = [
            supervisor_id
            for supervisor_id, supervisor in mailbox.supervisors.items()
            if interrupt_id in supervisor.interrupt_ids
        ]
        if len(owners) != 1:
            return False
        mailbox.offers[decision_id] = _Offer(
            decision=dict(decision),
            supervisor_id=owners[0],
        )
        while len(mailbox.offers) > MAX_PENDING_OFFERS:
            mailbox.offers.pop(next(iter(mailbox.offers)))
        return True


def commit(thread_id: str, decision: dict) -> bool:
    """Commit an offered decision and wake the owning asyncio supervisor."""
    decision_id = str((decision or {}).get("decision_id") or "")
    interrupt_id = str((decision or {}).get("interrupt_id") or "")
    if not thread_id or not decision_id or not interrupt_id:
        return False
    with _lock:
        mailbox = _mailboxes.get(thread_id)
        if not mailbox or not mailbox.active:
            return False
        if decision_id in mailbox.seen_commits:
            return True
        if interrupt_id in mailbox.seen_interrupts:
            return False
        offered = mailbox.offers.pop(decision_id, None)
        if offered is None:
            return False
        supervisor = mailbox.supervisors.get(offered.supervisor_id)
        if (
            supervisor is None
            or interrupt_id not in supervisor.interrupt_ids
        ):
            return False
        committed = {**offered.decision, **dict(decision)}
        supervisor.interrupt_ids.discard(interrupt_id)
        supervisor.commits[decision_id] = committed
        mailbox.seen_commits.add(decision_id)
        mailbox.seen_interrupts.add(interrupt_id)
        while len(mailbox.seen_commits) > MAX_SEEN_DECISIONS:
            mailbox.seen_commits.discard(next(iter(mailbox.seen_commits)))
        while len(mailbox.seen_interrupts) > MAX_SEEN_DECISIONS:
            mailbox.seen_interrupts.discard(next(iter(mailbox.seen_interrupts)))
        while len(supervisor.commits) > MAX_PENDING_COMMITS:
            supervisor.commits.pop(next(iter(supervisor.commits)))
        loop = supervisor.loop
        wakeup = supervisor.wakeup
    loop.call_soon_threadsafe(wakeup.set)
    return True


def _resolve_supervisor(
    mailbox: Optional[_Mailbox], supervisor_id: Optional[str],
) -> Optional[_Supervisor]:
    if mailbox is None:
        return None
    if supervisor_id:
        return mailbox.supervisors.get(supervisor_id)
    if len(mailbox.supervisors) == 1:
        return next(iter(mailbox.supervisors.values()))
    return None


def drain(thread_id: str, supervisor_id: Optional[str] = None) -> List[dict]:
    """Return committed decisions once, preserving transport arrival order."""
    if not thread_id:
        return []
    with _lock:
        mailbox = _mailboxes.get(thread_id)
        supervisor = _resolve_supervisor(mailbox, supervisor_id)
        if supervisor is None or not supervisor.commits:
            return []
        values = list(supervisor.commits.values())
        supervisor.commits.clear()
        supervisor.wakeup.clear()
        return values


async def wait(
    thread_id: str, supervisor_id: Optional[str] = None,
) -> List[dict]:
    """Wait until at least one committed decision is available."""
    while True:
        ready = drain(thread_id, supervisor_id)
        if ready:
            return ready
        with _lock:
            mailbox = _mailboxes.get(thread_id)
            supervisor = _resolve_supervisor(mailbox, supervisor_id)
            wakeup = supervisor.wakeup if supervisor else None
        if wakeup is None:
            return []
        await wakeup.wait()
