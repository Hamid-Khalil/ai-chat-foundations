# Apex Dynamics Information Security and Access Governance (2026)

## 1. Authentication Standards and Credential Hygiene
All corporate accounts and administrative bastions within Apex Dynamics enforce strict credential complexity:
- Passwords must be at least 16 characters in length and include uppercase, lowercase, numeric, and symbolic characters.
- Passwords expire every 90 calendar days; reuse of the last 12 historical passwords is strictly forbidden.
- Accounts are locked for 30 minutes after 5 consecutive failed authentication attempts.
- Multi-factor authentication (MFA) via FIDO2 hardware security keys is mandatory for all production node access. SMS-based MFA is prohibited under RFC-SEC-09.

## 2. Network Infrastructure and Remote Tunnels
Access to internal staging clusters and database replicas requires an active WireGuard VPN tunnel connecting to `gw-alpha.apex.internal` on port 51820.
- Workstation idle screens must engage an automated lock after 7 minutes of inactivity.
- Public Wi-Fi usage without VPN encapsulation is an automatic tier-one compliance violation.
- Corporate office Wi-Fi broadcasts exclusively on SSID `Apex-Secure-WPA3`. Guest network bandwidth is strictly throttled to 5 Mbps per device.
- Cryptographic keys for SSH authentication must be Ed25519 or 4096-bit RSA; DSA and ECDSA keys are permanently blocked at the gateway firewall.

## 3. Incident Response and Operations Hotline
Any suspected data exfiltration, lost hardware, or anomalous token activity must be declared within 20 minutes:
- Security Operations Center (SOC) emergency hotline: extension 5501.
- Internal Slack incident bridge: `#sec-emergency`.
- External incident coordinator: Marcus Vance (Node routing 0x9B4).

Third-party external penetration audits are conducted bi-annually during May and November under the supervision of the Chief Security Architect.
