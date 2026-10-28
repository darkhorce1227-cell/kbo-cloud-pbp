# KBO PBP automatic updater when GitHub-hosted Naver egress times out

## Verified diagnosis (2026-10-28 KST)

- Original Naver schedule endpoint and an actual completed inning relay returned HTTP 200 in roughly 0.1 seconds from the user's Windows computer on 2026-10-27.
- GitHub-hosted Ubuntu 24.04, Windows 2025 and macOS 15 all returned HTTP 000 / cURL timeout for both the 2026-10-26 Naver daily schedule and inning-1 relay; forced IPv4 also timed out. See Actions run **40617559193**.
- Changing only `runs-on: ubuntu-latest` to another GitHub-hosted OS is NOT a verified solution.
- The current verified rolling release contains 725 games through 2026-10-26. The 2026-10-27 game is unmerged as of this diagnosis.

## Prepared, but NOT ACTIVE until a runner is registered

`.github/workflows/update-pbp-local-windows.yml` provides a separate Naver-reachable Windows runner path.
The existing `.github/workflows/update-pbp.yml` is unchanged and retains all existing release/validation gates.

- Runs at 09:20 and 00:20 KST via GitHub schedule, or manual `workflow_dispatch`.
- For automatic runs, selects the last completed KST date (yesterday) and gives priority to unresolved dated recovery/audit targets.
- Uses exactly the same checked PBP program: `cloud/update_pbp_integrity.py`.
- Preserves source raw caches with `actions/cache`, runs binary-derived postgame score audit, stops on FAILED integrity and only publishes passing releases.
- Uses the same cross-workflow concurrency group, preventing release collisions.

**Important:** The first run on a new Windows runner can require downloading most of 2026's existing raw Naver relays to populate its cache. The 120-minute job cap is not a guarantee of completion. It also still needs verification that Python Requests from the runner account can reach Naver (successful interactive curl is encouraging but not definitive). Do not interpret a registered runner as a successfully updated release.

## Security first: current repo is PUBLIC

GitHub explicitly warns that self-hosted runners for public repositories are risky: workflow code can execute on the runner with its local permissions. **Do not register your primary personal/work PC to this public repository without hardening.** Prefer a separate isolated machine/VM and restricted user, or create a separate PRIVATE collection repo whose runner writes only audited data to the public project through a tightly scoped GitHub credential. Never publish the GitHub runner registration token in issues or ChatGPT.

See https://docs.github.com/en/actions/reference/security/secure-use

## One-time Windows runner prerequisites, on a dedicated isolated machine

1. Install **Git for Windows** (includes Git Bash) and **GitHub CLI**. In PowerShell with winget available:
   `winget install -e --id Git.Git`
   `winget install -e --id GitHub.cli`
2. In the GitHub repo owner settings, open **Settings → Actions → Runners → New self-hosted runner → Windows x64**. Follow GitHub's version-current PowerShell commands in **an Administrator PowerShell** at `C:\actions-runner`. Never reuse commands or expired registration tokens copied from an example.
3. During runner configuration, specify extra label **`kbo-naver`**, preserving default `self-hosted`, `Windows`, `X64` labels; optionally install as a Windows service so it starts on boot. To run unattended, keep the isolated machine online and awake at the scheduled times.
4. Verify `git`, `gh`, and the runner service are visible to the service environment. If GitHub CLI was installed after the runner service started, restart the service to refresh its PATH.
5. From **Actions → KBO PBP Auto Update via Local Windows Naver Runner → Run workflow**, run a known missing date with `target_date=2026-10-27`, then inspect all steps and the `latest-pbp` release. A passing workflow and updated SHA / 726 game IDs are necessary before declaring success.

GitHub runner installation docs: https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners

## Alternatives if a Windows PC cannot stay online

- A dedicated always-on host with confirmed Naver access can register a compatible runner (adapt workflow OS if not Windows); GitHub itself still handles scheduling, validation, caching and publishing.
- A private HTTPS collection relay on another network might make a purely GitHub-hosted job possible, but must first demonstrate Naver schedule **and** relay access; do not use unverified public proxies for source-of-truth sports data.

Do not modify or delete the last validated 725-game rolling release when upstream connectivity fails.
