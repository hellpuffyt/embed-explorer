# Security policy

## Supported versions

Only the latest commit on `main` receives fixes. There are no maintained release branches.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's advisory form:
https://github.com/hellpuffyt/embed-explorer/security/advisories/new

Do not open a public issue for a suspected vulnerability. Include the version or commit,
a minimal reproduction, and the impact you observed. You should receive an acknowledgement
within a few days; fixes are released as soon as they are verified.

## Scope

embed-explorer reads local .npy/.jsonl/.csv files (numpy with allow_pickle=False) and prints results; no network access. Relevant reports include: a crafted input file that executes code, writes files, or exhausts memory beyond the documented O(n^2) exact search.
