from app.sandbox import SandboxConfig, ContainerManager
import time

_sandbox_config = SandboxConfig()
_container_manager = ContainerManager(_sandbox_config)
_container_manager.cleanup_stale()

session_id = 'test_sandbox'

start_time = time.time()
_container_manager.get_or_create(session_id)
print(f'get docker contain, cost: {time.time() - start_time}')

while True:
    cmd = input()
    if cmd == 'exit':
        break

_container_manager.destroy(session_id)
print("exit...")