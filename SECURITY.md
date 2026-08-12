# Security Policy

## Supported Versions

Security fixes are applied to the latest revision of the default branch.

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting feature when available, or contact the
repository owner privately through the contact information on their GitHub
profile.

Include the affected component, reproduction steps, expected impact, and any
proposed mitigation. You should receive an acknowledgement within seven days.

SchedForge executes generated native code. Treat untrusted IR, schedules, and
cache artifacts as untrusted input; do not execute them in a privileged process.
