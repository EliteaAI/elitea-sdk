"""Tests for elitea_sdk.tools.utils utilities."""
import pytest

from elitea_sdk.tools.utils import normalize_pem_key


TEST_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAmY8p/k5MQWFxY+BKBc3B6bTIcdd879KZUh+G0H44xRDn/ds9
HH10geU1vLp7zMn+RgPepvZ+3qcJcAjV/3SUhzDSenSg3ziwMwhp71AHi4qnNIz+
J9mickGA8KiZP1JqQ7iYytOG5tMa5BvK+TbvsZuAY2R2wMZlgYhvIvoCxTkDhNue
q/tFbb33vN4CbxGFfHeBQRR9FwaiQMqTPjFj8YDxOPLdeSOiCyGvur16gIqPoE0f
z8IqFu7VXVPE88yzzFZs2jflBHXcHOHOY5iEvQWPxfOMks9XJ91sxsVl2yLpNe0T
XoWl4djHKHGcaAktG0f9NEDOtZPsGYbkDqutHQIDAQABAoIBABiJIv0StqV1s9/w
/PLbcwHsFGD4POq23C+egPR8TobSUjciGiwcsYp1vLXYmCJbHglC4gcMeK4Lw3rG
tqa4EmldeCv/yZqRHLoyvHZz85iswVWot41XfVjDNZ5+WeofuYHzk1pJHRNxpKjQ
9ggG6pqWzZdT/JOFv79INPXCO8IWQBiO/knZpm7OkfwotkdwPWTWJD+rXTTzG4+x
6CuyjtSsPfHFg8spDYS0DS9++ZtnFSP5GipuyMQn6byLKK6h1tpXQYej2XEKIfnk
dArDctXR1H/h7MJRBRkSstqtNvTZit+yG/LQTejCFRP9ZKLuoS6k+idnnvemgxyL
whH5+YECgYEAz2QfIxwFV1boNDR/I4K6PFVyce4RWA7YT1eDCFRI0Z0DfFl+iCWE
lHoumZiw9AYJk0qJF8qzupewzAWp+N0dc/6Wrweuiyfj+Z2KEL1yiXK31JlQv4R9
4/RhgVev5mpRhe3V0oeI+h7kiu1lvSIIPOu+MckOAsymkzFlTD+mysECgYEAvY0F
/9zflyr3oN9++LxjGRAM92tfC4zZS72kTIvLAHvluLB62/EETzNy+RnFA7A8Aazm
+bFZmyxyd2SfdnRtYFazPWc6XzsCDHLKb91WoRwbPVngpjQIlj56ZLt3yPPB+wUt
0Ir/osaPayfOmqd00/E+HFR661FqWioUvjvuRV0CgYEApM57r/rkg7Occ2AEaMPK
G4gLml4FimTBoMt6ZXQVKf4MdxTnaGnoIdW2kni0pjbmBRaGO1Hp4a4J4RffKtUM
QtFeDVmdaxgYIpT+0q66BmATle8ALDGtmSjrE04Lip+SiUunT9ZFE/7Yv05IOzSA
N2lfi1Cqwa6/8NigFye99AECgYAKVYbvNSaHglMv1R+CBhtNAYADeTocUhiCtZsg
hTqTEy4qDI0WMqSugLqS9CG2msQav0d0c4PUHu86rSS4e45/AxsQjPE0we3Rqex5
ftK7Q+IETUMfLJUPQ+a+WS4lqYx42AZwaTOYt0SYbfoomlqXN37QYpa0/6JRuhuZ
Z4ENDQKBgCmSkMoEwCfYmirGXlSz3bm8SRBRuw6rigy2+9eeyfAE1MgPYVYTj9Ba
kYWNdLCJdoN9db6x/5J+5vFsCKrooFxdNLdT8y/IXwN7Toau6oWfvAPey4ecIuwQ
PzbRu77QKe3K92EMPjzNCR6zZi/l7Wdw1UUjHXuUDXCkSkfk8siL
-----END RSA PRIVATE KEY-----"""

TEST_KEY_BODY = """MIIEowIBAAKCAQEAmY8p/k5MQWFxY+BKBc3B6bTIcdd879KZUh+G0H44xRDn/ds9
HH10geU1vLp7zMn+RgPepvZ+3qcJcAjV/3SUhzDSenSg3ziwMwhp71AHi4qnNIz+
J9mickGA8KiZP1JqQ7iYytOG5tMa5BvK+TbvsZuAY2R2wMZlgYhvIvoCxTkDhNue
q/tFbb33vN4CbxGFfHeBQRR9FwaiQMqTPjFj8YDxOPLdeSOiCyGvur16gIqPoE0f
z8IqFu7VXVPE88yzzFZs2jflBHXcHOHOY5iEvQWPxfOMks9XJ91sxsVl2yLpNe0T
XoWl4djHKHGcaAktG0f9NEDOtZPsGYbkDqutHQIDAQABAoIBABiJIv0StqV1s9/w
/PLbcwHsFGD4POq23C+egPR8TobSUjciGiwcsYp1vLXYmCJbHglC4gcMeK4Lw3rG
tqa4EmldeCv/yZqRHLoyvHZz85iswVWot41XfVjDNZ5+WeofuYHzk1pJHRNxpKjQ
9ggG6pqWzZdT/JOFv79INPXCO8IWQBiO/knZpm7OkfwotkdwPWTWJD+rXTTzG4+x
6CuyjtSsPfHFg8spDYS0DS9++ZtnFSP5GipuyMQn6byLKK6h1tpXQYej2XEKIfnk
dArDctXR1H/h7MJRBRkSstqtNvTZit+yG/LQTejCFRP9ZKLuoS6k+idnnvemgxyL
whH5+YECgYEAz2QfIxwFV1boNDR/I4K6PFVyce4RWA7YT1eDCFRI0Z0DfFl+iCWE
lHoumZiw9AYJk0qJF8qzupewzAWp+N0dc/6Wrweuiyfj+Z2KEL1yiXK31JlQv4R9
4/RhgVev5mpRhe3V0oeI+h7kiu1lvSIIPOu+MckOAsymkzFlTD+mysECgYEAvY0F
/9zflyr3oN9++LxjGRAM92tfC4zZS72kTIvLAHvluLB62/EETzNy+RnFA7A8Aazm
+bFZmyxyd2SfdnRtYFazPWc6XzsCDHLKb91WoRwbPVngpjQIlj56ZLt3yPPB+wUt
0Ir/osaPayfOmqd00/E+HFR661FqWioUvjvuRV0CgYEApM57r/rkg7Occ2AEaMPK
G4gLml4FimTBoMt6ZXQVKf4MdxTnaGnoIdW2kni0pjbmBRaGO1Hp4a4J4RffKtUM
QtFeDVmdaxgYIpT+0q66BmATle8ALDGtmSjrE04Lip+SiUunT9ZFE/7Yv05IOzSA
N2lfi1Cqwa6/8NigFye99AECgYAKVYbvNSaHglMv1R+CBhtNAYADeTocUhiCtZsg
hTqTEy4qDI0WMqSugLqS9CG2msQav0d0c4PUHu86rSS4e45/AxsQjPE0we3Rqex5
ftK7Q+IETUMfLJUPQ+a+WS4lqYx42AZwaTOYt0SYbfoomlqXN37QYpa0/6JRuhuZ
Z4ENDQKBgCmSkMoEwCfYmirGXlSz3bm8SRBRuw6rigy2+9eeyfAE1MgPYVYTj9Ba
kYWNdLCJdoN9db6x/5J+5vFsCKrooFxdNLdT8y/IXwN7Toau6oWfvAPey4ecIuwQ
PzbRu77QKe3K92EMPjzNCR6zZi/l7Wdw1UUjHXuUDXCkSkfk8siL"""

TEST_PKCS8_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCZjyn+TkxBYXFj
4EoFzcHptMhx13zv0plSH4bQfjjFEOf92z0cfXSB5TW8unvMyf5GA96m9n7epwlw
CNX/dJSHMNJ6dKDfOLAzCGnvUAeLiqc0jP4n2aJyQYDwqJk/UmpDuJjK04bm0xrk
G8r5Nu+xm4BjZHbAxmWBiG8i+gLFOQOE256r+0Vtvfe83gJvEYV8d4FBFH0XBqJA
ypM+MWPxgPE48t15I6ILIa+6vXqAio+gTR/PwioW7tVdU8TzzLPMVmzaN+UEddwc
4c5jmIS9BY/F84ySz1cn3WzGxWXbIuk17RNehaXh2McocZxoCS0bR/00QM61k+wZ
huQOq60dAgMBAAECggEAGIki/RK2pXWz3/D88ttzAewUYPg86rbcL56A9HxOhtJS
NyIaLByxinW8tdiYIlseCULiBwx4rgvDesa2prgSaV14K//JmpEcujK8dnPzmKzB
Vai3jVd9WMM1nn5Z6h+5gfOTWkkdE3GkqND2CAbqmpbNl1P8k4W/v0g09cI7whZA
GI7+Sdmmbs6R/Ci2R3A9ZNYkP6tdNPMbj7HoK7KO1Kw98cWDyykNhLQNL375m2cV
I/kaKm7IxCfpvIsorqHW2ldBh6PZcQoh+eR0CsNy1dHUf+HswlEFGRKy2q029NmK
37Ib8tBN6MIVE/1kou6hLqT6J2ee96aDHIvCEfn5gQKBgQDPZB8jHAVXVug0NH8j
gro8VXJx7hFYDthPV4MIVEjRnQN8WX6IJYSUei6ZmLD0BgmTSokXyrO6l7DMBan4
3R1z/pavB66LJ+P5nYoQvXKJcrfUmVC/hH3j9GGBV6/malGF7dXSh4j6HuSK7WW9
Igg8674xyQ4CzKaTMWVMP6bKwQKBgQC9jQX/3N+XKveg3374vGMZEAz3a18LjNlL
vaRMi8sAe+W4sHrb8QRPM3L5GcUDsDwBrOb5sVmbLHJ3ZJ92dG1gVrM9ZzpfOwIM
cspv3VahHBs9WeCmNAiWPnpku3fI88H7BS3Qiv+ixo9rJ86ap3TT8T4cVHrrUWpa
KhS+O+5FXQKBgQCkznuv+uSDs5xzYARow8obiAuaXgWKZMGgy3pldBUp/gx3FOdo
aegh1baSeLSmNuYFFoY7UenhrgnhF98q1QxC0V4NWZ1rGBgilP7SrroGYBOV7wAs
Ma2ZKOsTTguKn5KJS6dP1kUT/ti/Tkg7NIA3aV+LUKrBrr/w2KAXJ730AQKBgApV
hu81JoeCUy/VH4IGG00BgAN5OhxSGIK1myCFOpMTLioMjRYypK6AupL0IbaaxBq/
R3Rzg9Qe7zqtJLh7jn8DGxCM8TTB7dGp7Hl+0rtD4gRNQx8slQ9D5r5ZLiWpjHjY
BnBpM5i3RJht+iiaWpc3ftBilrT/olG6G5lngQ0NAoGAKZKQygTAJ9iaKsZeVLPd
ubxJEFG7DquKDLb7157J8ATUyA9hVhOP0FqRhY10sIl2g311vrH/kn7m8WwIquig
XF00t1PzL8hfA3tOhq7qhZ+8A97Lh5wi7BA/NtG7vtAp7cr3YQw+PM0JHrNmL+Xt
Z3DVRSMde5QNcKRKR+TyyIs=
-----END PRIVATE KEY-----"""


class TestNormalizePemKey:
    """Test normalize_pem_key utility function."""

    def test_pkcs1_key_with_headers(self):
        """PKCS#1 key with headers should be preserved."""
        normalized = normalize_pem_key(TEST_PRIVATE_KEY)
        assert normalized.startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert normalized.endswith("-----END RSA PRIVATE KEY-----")
        assert "-----BEGIN PRIVATE KEY-----" not in normalized

    def test_pkcs8_key_with_headers(self):
        """PKCS#8 key with headers should be preserved."""
        normalized = normalize_pem_key(TEST_PKCS8_KEY)
        assert normalized.startswith("-----BEGIN PRIVATE KEY-----")
        assert normalized.endswith("-----END PRIVATE KEY-----")
        assert "-----BEGIN RSA PRIVATE KEY-----" not in normalized

    def test_key_body_without_headers(self):
        """Key body without headers should default to PKCS#1."""
        normalized = normalize_pem_key(TEST_KEY_BODY)
        assert normalized.startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert normalized.endswith("-----END RSA PRIVATE KEY-----")

    def test_single_line_pkcs1_key(self):
        """Single-line PKCS#1 key should be normalized to multi-line."""
        single_line = TEST_PRIVATE_KEY.replace("\n", " ")
        normalized = normalize_pem_key(single_line)
        assert normalized.startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert normalized.endswith("-----END RSA PRIVATE KEY-----")
        assert "\n" in normalized

    def test_single_line_pkcs8_key(self):
        """Single-line PKCS#8 key should be normalized to multi-line."""
        single_line = TEST_PKCS8_KEY.replace("\n", " ")
        normalized = normalize_pem_key(single_line)
        assert normalized.startswith("-----BEGIN PRIVATE KEY-----")
        assert normalized.endswith("-----END PRIVATE KEY-----")
        assert "\n" in normalized

    def test_footer_only_pkcs1(self):
        """Key with only footer (no header) should be sanitized.

        Bug #3986: Footer-only case was not sanitized, leaving END marker in key body.
        """
        key_with_footer_only = TEST_KEY_BODY + "\n-----END RSA PRIVATE KEY-----"
        normalized = normalize_pem_key(key_with_footer_only)
        assert normalized.startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert normalized.endswith("-----END RSA PRIVATE KEY-----")
        assert normalized.count("-----END RSA PRIVATE KEY-----") == 1

    def test_footer_only_pkcs8(self):
        """Key with only PKCS#8 footer should be sanitized."""
        key_with_footer_only = TEST_KEY_BODY + "\n-----END PRIVATE KEY-----"
        normalized = normalize_pem_key(key_with_footer_only)
        assert normalized.startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert normalized.endswith("-----END RSA PRIVATE KEY-----")
        assert "-----END PRIVATE KEY-----" not in normalized
