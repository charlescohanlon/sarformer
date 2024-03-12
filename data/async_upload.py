from fileinput import filename
import os
import time
from ms_graph import generate_access_token, GRAPH_API_ENDPOINT
from dotenv import load_dotenv

import asyncio
import aiohttp
from aiohttp.client import ClientSession
from tenacity import *

load_dotenv()

# app id may or may not work since it's attached to logan's app
APP_ID = os.environ.get("APP_ID")
SCOPES = ["Files.ReadWrite"]

# path where tif imgs should be put
SRC_PATH = f"/Users/loganbarker/Desktop/S&R/AI4SR/onedrive/test_imgs"

POLL_PERIOD = 0.001
TIMEOUT_DURATION = 120

# folder must be in root of onedrive
DEST_FOLDER_NAME = ""

access_token = generate_access_token(APP_ID, SCOPES)
headers = {"Authorization": "Bearer " + access_token["access_token"]}


async def upload_tif(file_path: str, session: ClientSession):
    with open(file_path, "rb") as upload:
        media_content = upload.read()
    file_name = os.path.basename(file_path)
    async with session.put(
        GRAPH_API_ENDPOINT
        + f"/me/drive/items/root:/{DEST_FOLDER_NAME}/{file_name}:/content",
        headers=headers,
        data=media_content,
    ) as response:
        result = await response.text()
        print(result)


# https://skillshats.com/blogs/send-http-requests-as-fast-as-possible-in-python/#:~:text=Asyncio%20is%20so%20fast%20that,your%20machine%20and%20internet%20bandwidth.
@retry(
    reraise=True,
    wait=wait_fixed(10),
    retry=retry_if_exception_type(aiohttp.ServerDisconnectedError),
    stop=stop_after_attempt(10),
)
async def main():
    # limit to 10 TCP connections to reuse in a session (any more and we might get banned for DOSing)
    conn = aiohttp.TCPConnector(limit=10)

    should_exit = False
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        while True:
            elapsed_wait_time = 0

            while file_names := os.listdir(SRC_PATH) == 0:
                time.sleep(POLL_PERIOD)

                # exits after no new files are added to the src folder after TIMEOUT_PERIOD seconds
                elapsed_wait_time += POLL_PERIOD
                if elapsed_wait_time >= TIMEOUT_DURATION:
                    # use flag instead of return b/c needs to await tasks outside loop but within session
                    should_exit = True
                    break
            if should_exit:
                break

            file_paths = [SRC_PATH + "/" + name for name in file_names]
            for path in file_paths:
                tasks.append(
                    asyncio.create_task(upload_tif(file_path=path, session=session))
                )
                # delete tif after it's been sent off
                os.remove(path)

        # the await must be nest inside of the session
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
