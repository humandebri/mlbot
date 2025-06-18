#!/usr/bin/env python3
"""Test Python 3.13 async method issue"""

import asyncio


class TestClass:
    def normal_method(self):
        print("Normal method called")
    
    async def async_method(self):
        print("Async method called")
        await asyncio.sleep(0.1)


async def main():
    obj = TestClass()
    
    # Call normal method
    obj.normal_method()
    
    # Call async method
    await obj.async_method()
    
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())