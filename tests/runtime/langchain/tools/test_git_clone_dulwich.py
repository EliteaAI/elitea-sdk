"""#7205: git.clone end-to-end against real local git servers on dulwich >= 1.2.

dulwich 1.2 dropped ``dulwich.contrib`` (the paramiko SSH vendor the SDK used) and
moved ``Repo.reset_index`` to ``WorkTree``. These tests exercise the real
``elitea_sdk.runtime.langchain.tools.git.clone`` code path over:

* local filesystem source (branch present / missing -> default branch fallback,
  ``delete_git_dir``, upstream tracking config),
* smart HTTP via ``git http-backend`` behind a CGI handler,
* SSH via a throwaway ``sshd`` on a high port with a freshly generated RSA key,
  passed both as ``key_data`` (inline, ``|``-separated, as the UI sends it) and
  as ``key_filename``.

Server-backed tests skip when ``git``/``git-http-backend``/``sshd`` are missing.
No network is used beyond 127.0.0.1.
"""

import http.server
import os
import shutil
import socket
import subprocess
import threading
import time

import pytest

dulwich = pytest.importorskip("dulwich")

# The generic PR job overlays this checkout onto a fixed dependency image without
# installing ``.[runtime]``, so it may still carry the pre-1.2 dulwich. These tests
# target the pinned version and run wherever it is installed.
if dulwich.__version__ < (1, 2, 9):
    pytest.skip(
        f"requires dulwich >= 1.2.9 (found {'.'.join(map(str, dulwich.__version__))})",
        allow_module_level=True,
    )

from elitea_sdk.runtime.langchain.tools import git as sdk_git  # noqa: E402

GIT = shutil.which("git")
SSHD = shutil.which("sshd") or ("/usr/sbin/sshd" if os.path.exists("/usr/sbin/sshd") else None)


def _git_http_backend():
    try:
        exec_path = subprocess.run(
            [GIT, "--exec-path"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # pylint: disable=W0703
        return None
    candidate = os.path.join(exec_path, "git-http-backend")
    return candidate if os.path.exists(candidate) else None


requires_git = pytest.mark.skipif(GIT is None, reason="git binary not available")


def _run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module", autouse=True)
def _patches():
    # The loaders call clone() without apply_patches(); exercise both states by
    # applying here, which also proves the patches still bind on dulwich 1.2.
    sdk_git.apply_patches()


@pytest.fixture(scope="module")
def source_repo(tmp_path_factory):
    """Bare repo with `main` (a.txt) and `feat` (a.txt + b.txt)."""
    if GIT is None:
        pytest.skip("git binary not available")
    root = tmp_path_factory.mktemp("src")
    work = root / "work"
    work.mkdir()
    _run(GIT, "init", "-q", "-b", "main", cwd=work)
    _run(GIT, "config", "user.email", "t@t", cwd=work)
    _run(GIT, "config", "user.name", "t", cwd=work)
    (work / "a.txt").write_text("one\n")
    _run(GIT, "add", ".", cwd=work)
    _run(GIT, "commit", "-q", "-m", "c1", cwd=work)
    _run(GIT, "checkout", "-q", "-b", "feat", cwd=work)
    (work / "b.txt").write_text("two\n")
    _run(GIT, "add", ".", cwd=work)
    _run(GIT, "commit", "-q", "-m", "c2", cwd=work)
    _run(GIT, "checkout", "-q", "main", cwd=work)
    bare = root / "repo.git"
    _run(GIT, "clone", "-q", "--bare", str(work), str(bare))
    _run(GIT, "config", "http.receivepack", "false", cwd=bare)
    return bare


def _files(path):
    return sorted(p for p in os.listdir(path) if p != ".git")


def _assert_branch_checkout(target, branch, expected_files):
    assert _files(target) == expected_files
    repo = dulwich.repo.Repo(str(target))
    assert repo.refs.read_ref(b"HEAD") == b"ref: refs/heads/" + branch.encode()
    config = repo.get_config()
    assert config.get((b"branch", branch.encode()), b"remote") == b"origin"
    assert config.get((b"branch", branch.encode()), b"merge") == b"refs/heads/" + branch.encode()


@requires_git
@pytest.mark.parametrize("branch,files", [("main", ["a.txt"]), ("feat", ["a.txt", "b.txt"])])
def test_local_clone_checks_out_requested_branch(source_repo, tmp_path, branch, files):
    target = tmp_path / "clone"
    sdk_git.clone(str(source_repo), str(target), branch=branch)
    _assert_branch_checkout(target, branch, files)


@requires_git
def test_local_clone_missing_branch_falls_back_to_default(source_repo, tmp_path):
    target = tmp_path / "clone"
    sdk_git.clone(str(source_repo), str(target), branch="does-not-exist")
    # Fallback path: default branch tree checked out and tracked.
    _assert_branch_checkout(target, "main", ["a.txt"])


@requires_git
def test_local_clone_delete_git_dir(source_repo, tmp_path):
    target = tmp_path / "clone"
    sdk_git.clone(str(source_repo), str(target), branch="feat", delete_git_dir=True)
    assert sorted(os.listdir(target)) == ["a.txt", "b.txt"]


# --- smart HTTP -------------------------------------------------------------


@pytest.fixture(scope="module")
def http_url(source_repo):
    backend = _git_http_backend()
    if backend is None:
        pytest.skip("git-http-backend not available")
    project_root = str(source_repo.parent)

    class Handler(http.server.CGIHTTPRequestHandler):
        def log_message(self, *args):  # silence test output
            pass

        def _serve(self):
            path, _, query = self.path.partition("?")
            env = dict(os.environ)
            env.update({
                "GIT_PROJECT_ROOT": project_root,
                "GIT_HTTP_EXPORT_ALL": "1",
                "REQUEST_METHOD": self.command,
                "PATH_INFO": path,
                "QUERY_STRING": query,
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                "REMOTE_ADDR": "127.0.0.1",
            })
            if self.headers.get("Git-Protocol"):
                env["GIT_PROTOCOL"] = self.headers["Git-Protocol"]
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            proc = subprocess.run([backend], input=body, env=env, capture_output=True, check=False)
            head, _, payload = proc.stdout.partition(b"\r\n\r\n")
            if not _:
                head, _, payload = proc.stdout.partition(b"\n\n")
            status = 200
            headers = []
            for line in head.decode("latin-1").splitlines():
                key, _, value = line.partition(":")
                if key.lower() == "status":
                    status = int(value.strip().split()[0])
                elif key:
                    headers.append((key, value.strip()))
            self.send_response(status)
            for key, value in headers:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _serve
        do_POST = _serve

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/{source_repo.name}"
    server.shutdown()


@requires_git
@pytest.mark.parametrize("branch,files", [("main", ["a.txt"]), ("feat", ["a.txt", "b.txt"])])
def test_http_clone(http_url, tmp_path, branch, files):
    target = tmp_path / "clone"
    sdk_git.clone(http_url, str(target), branch=branch)
    _assert_branch_checkout(target, branch, files)


@requires_git
def test_http_clone_honors_ssl_cert_file_patch(http_url, tmp_path, monkeypatch, source_repo):
    # Exercises patched HttpGitClient.from_parsedurl config injection on dulwich 1.2.
    ca = tmp_path / "ca.pem"
    ca.write_text("")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    target = tmp_path / "clone"
    sdk_git.clone(http_url, str(target), branch="main")
    assert _files(target) == ["a.txt"]


# --- SSH (paramiko vendor) --------------------------------------------------


@pytest.fixture(scope="module")
def ssh_server(source_repo, tmp_path_factory):
    if SSHD is None:
        pytest.skip("sshd not available")
    root = tmp_path_factory.mktemp("sshd")
    host_key = root / "host_rsa"
    user_key = root / "user_rsa"
    _run("ssh-keygen", "-q", "-t", "rsa", "-b", "2048", "-N", "", "-m", "PEM", "-f", str(host_key))
    _run("ssh-keygen", "-q", "-t", "rsa", "-b", "2048", "-N", "", "-m", "PEM", "-f", str(user_key))
    authorized = root / "authorized_keys"
    shutil.copy(str(user_key) + ".pub", authorized)
    os.chmod(authorized, 0o600)
    port = _free_port()
    config = root / "sshd_config"
    config.write_text("\n".join([
        f"Port {port}",
        "ListenAddress 127.0.0.1",
        f"HostKey {host_key}",
        f"PidFile {root / 'sshd.pid'}",
        f"AuthorizedKeysFile {authorized}",
        "PasswordAuthentication no",
        "KbdInteractiveAuthentication no",
        "PubkeyAuthentication yes",
        "UsePAM no",
        "StrictModes no",
        "",
    ]))
    proc = subprocess.Popen(
        [SSHD, "-D", "-e", "-f", str(config)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.skip(f"sshd failed to start: {proc.stderr.read().decode(errors='replace')[:300]}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.skip("sshd did not start listening")
    user = os.environ.get("USER") or subprocess.run(
        ["id", "-un"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    yield {
        "url": f"ssh://{user}@127.0.0.1:{port}{source_repo}",
        "key_file": str(user_key),
        "key_data": user_key.read_text().strip().replace("\n", "|"),
    }
    proc.kill()
    proc.wait(timeout=5)


@requires_git
@pytest.mark.parametrize("branch,files", [("main", ["a.txt"]), ("feat", ["a.txt", "b.txt"])])
def test_ssh_clone_with_inline_key_data(ssh_server, tmp_path, branch, files):
    target = tmp_path / "clone"
    sdk_git.clone(ssh_server["url"], str(target), branch=branch, key_data=ssh_server["key_data"])
    _assert_branch_checkout(target, branch, files)


@requires_git
def test_ssh_clone_with_key_filename(ssh_server, tmp_path):
    target = tmp_path / "clone"
    sdk_git.clone(ssh_server["url"], str(target), branch="feat", key_filename=ssh_server["key_file"])
    _assert_branch_checkout(target, "feat", ["a.txt", "b.txt"])


def test_ssh_vendor_is_sdk_owned_after_patches():
    # dulwich >= 1.2 ships no contrib vendor; the SDK must provide its own.
    assert dulwich.client.get_ssh_vendor is sdk_git.ParamikoSSHVendor
