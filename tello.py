from contextlib import asynccontextmanager
from typing import (
    AsyncIterator,
    Awaitable,
    TypeAlias,
    Callable,
)

import asyncio
import cv2
import av


TELLO_IP = '192.168.10.1'

VIDEO_PORT   = 11111
STATE_PORT   = 8888
COMMAND_PORT = 8889


class Protocol(asyncio.Protocol):
    __slots__ = ('queue',)

    def __init__(self) -> None:
        self.queue = asyncio.Queue[bytes]()

    def connection_made(self, _) -> None:
        print('Connection Made')

    def datagram_received(self, data: bytes, _) -> None:
        self.queue.put_nowait(data)

    def error_received(self, exc: Exception) -> None:
        raise exc

    def connection_lost(self, exc: Exception | None) -> None:
        print('Connection Lost')
        
        if exc is not None:
            raise exc


def timeout(*, length=5.0, retries=0):
    def decorator(fn: Callable):
        async def wrapper(*args, **kwargs) -> Callable:
            for _ in range(retries):
                try:
                    async with asyncio.timeout(length):
                        return await fn(*args, **kwargs)
                
                except asyncio.TimeoutError:
                    continue

            raise asyncio.TimeoutError(f'{fn.__name__} timed out after max retries: {retries}')

        return wrapper
    return decorator


class Drone:
    Send: TypeAlias = Callable[[bytes], None]
    Recv: TypeAlias  = Callable[[], Awaitable[bytes]]

    __slots__ = ('_send', '_recv')

    def __init__(self, _send: Send, _recv: Recv) -> None:
        self._send = _send
        self._recv = _recv

    async def send(self, msg: str) -> str:
        self._send(msg.encode('utf-8'))
        resp = await self._recv()

        await asyncio.sleep(.1)

        try:
            return resp.decode()

        except UnicodeDecodeError:
            return None

    async def stream(self) -> AsyncIterator:
        await self.send('streamon')
        await asyncio.sleep(1)

        container = av.open('udp://@0.0.0.0:11111')
    
        try:
            for frame in container.decode(video=0):
                frame = frame.to_ndarray(format='bgr24')
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame

        finally:
            await self.send('streamoff')


@asynccontextmanager
async def conn(ip=TELLO_IP) -> AsyncIterator:
    loop = asyncio.get_running_loop()

    cmd_transport, cmd_protocol = await loop.create_datagram_endpoint(
        Protocol,
        remote_addr=(ip, COMMAND_PORT),
    )

    def send(data: bytes) -> None:
        cmd_transport.sendto(data)

    @timeout(retries=4)
    async def recv() -> bytes:
        return await cmd_protocol.queue.get()

    async def keepalive() -> None:
        while True:
            await asyncio.sleep(10)
        
            if cmd_protocol.queue.empty():
                print('Keepalive')
                send('keepalive')
                await cmd_protocol.queue.get()

    #task = asyncio.create_task(keepalive())

    try:
        yield Drone(send, recv) 

    except Exception as e:
        print('Ran into an error during drone connection')
        raise e

    finally:
        #task.cancel()
        cmd_transport.close()


    '''
    async def stream(self, *, process) -> AsyncIterator:
        queue = asyncio.Queue(maxsize=10)

        async def batch():
            while True:
                batch      = [await queue.get()]
                batch_task = asyncio.create_task(process(batch))

                while not queue.empty() and not batch_task.done():
                    print('Adding frame to batch')
                    batch.append(await queue.get())

                yield await batch_task
                
        async def fetch():
            container = av.open('udp://@0.0.0.0:11111')
            for frame in container.decode(video=0):
                try:
                    await asyncio.wait_for(queue.put(frame), timeout=.2)
                    print('Putting frame in queue')
                
                except asyncio.TimeoutError:
                    print('Dropping frame')

        fetch_task = asyncio.create_task(fetch())

        try:
            async for pred in batch():
                yield pred

        finally:
            fetch_task.cancel()
            await self.send('streamoff')
'''