# Governance

pyTOST uses a maintainer-led model. Dylan Hettinger
([@dhetting](https://github.com/dhetting)) is the current maintainer and is responsible
for scope, technical decisions, pull-request approval, and releases.

## Proposals and decisions

Anyone may report a problem, propose a change, or contribute through public GitHub
issues and pull requests. Discuss new inference methods, public API changes, and
dependency changes in an issue before implementing them.

The maintainer considers scientific justification, reproducible evidence, effects on
existing users, and maintenance cost. Decisions are made through public discussion,
not a vote; the maintainer makes the final decision and records its rationale in the
relevant issue or pull request. Contributors may request reconsideration with new
evidence. Sensitive security reports and conduct matters are handled privately.

## Review and releases

Changes should include appropriate tests and documentation. Changes to statistical
methods should explain their assumptions and include validation suited to the claim.
Passing CI is necessary but does not by itself establish statistical validity.
The maintainer reviews contributions before merging and is responsible for release
versioning and notes in [CHANGELOG.md](CHANGELOG.md). Breaking public API changes are
announced there before release, as described in the README.

## Participation and continuity

Participation follows [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Contribution routes
are described in [CONTRIBUTING.md](CONTRIBUTING.md), and support expectations in
[SUPPORT.md](SUPPORT.md). Contributions do not confer an obligation to maintain the
project.

Additional maintainers may be appointed by the current maintainer with the
candidate's agreement, following sustained contributions and review participation.
Appointments and any transfer of maintenance are announced publicly and reflected
in this file. There is currently no formal succession arrangement or guaranteed
maintenance term.

Changes to this governance policy are proposed through a public issue or pull
request and approved by the maintainer.
