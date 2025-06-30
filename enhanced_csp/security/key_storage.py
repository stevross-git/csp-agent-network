import keyring
from keyring.errors import NoKeyringError
from cryptography.fernet import Fernet
import time

class SecureKeyManager:
    """Store node private keys encrypted in the OS keyring."""

    def store_node_key(self, node_id: str, private_key: bytes) -> None:
        encryption_key = Fernet.generate_key()
        f = Fernet(encryption_key)
        encrypted = f.encrypt(private_key)
        try:
            keyring.set_password("enhanced_csp", f"node_{node_id}", encrypted.decode())
            keyring.set_password("enhanced_csp", f"enc_{node_id}", encryption_key.decode())
            keyring.set_password("enhanced_csp", f"rot_{node_id}", str(int(time.time())))
        except NoKeyringError:
            pass

    def load_node_key(self, node_id: str) -> bytes | None:
        try:
            encrypted = keyring.get_password("enhanced_csp", f"node_{node_id}")
            enc_key = keyring.get_password("enhanced_csp", f"enc_{node_id}")
        except NoKeyringError:
            return None

        if not encrypted or not enc_key:
            return None
        f = Fernet(enc_key.encode())
        return f.decrypt(encrypted.encode())

    def should_rotate(self, node_id: str, max_age_days: int = 30) -> bool:
        try:
            ts = keyring.get_password("enhanced_csp", f"rot_{node_id}")
        except NoKeyringError:
            return False
        if not ts:
            return False
        try:
            ts_int = int(ts)
        except ValueError:
            return False
        return (time.time() - ts_int) > (max_age_days * 86400)
