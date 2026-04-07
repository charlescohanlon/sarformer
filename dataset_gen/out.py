import os
import time
from ms_graph import generate_access_token, GRAPH_API_ENDPOINT
from dotenv import load_dotenv

import requests
import threading as t, queue

load_dotenv()

# app id may or may not work since it's attached to logan's app
APP_ID = os.environ.get("APP_ID")
SCOPES = ["Files.ReadWrite"]

# path where tif imgs should be put
SRC_FOLDER = "/home/charles/AI4SAR/dataset_gen/data"

POLL_PERIOD = 1  # in seconds
TIMEOUT_DURATION = 600  # in seconds

# path must be from root of onedrive
DEST_FOLDER_NAME = "ArcGIS_Data"

access_token = generate_access_token(APP_ID, SCOPES)
headers = {"Authorization": "Bearer " + access_token["access_token"]}

NUM_PRODUCERS = 1
NUM_CONSUMERS = 100

# TODO: figure out how to extend msal token expiration


def producer(queue):
    diff = {}
    while True:
        elapsed_wait_time = 0

        while len(file_names := os.listdir(SRC_FOLDER)) == 0:
            time.sleep(POLL_PERIOD)
            elapsed_wait_time += POLL_PERIOD
            if elapsed_wait_time >= TIMEOUT_DURATION:
                return

        for name in file_names:
            if name not in diff.keys():
                queue.put(name)
                diff[name] = "1"


def consumer(queue):
    while True:
        elapsed_wait_time = 0
        while queue.empty():
            time.sleep(POLL_PERIOD)
            elapsed_wait_time += POLL_PERIOD
            if elapsed_wait_time >= TIMEOUT_DURATION:
                return

        file_name = queue.get()
        src_path = SRC_FOLDER + "/" + file_name
        with open(src_path, "rb") as upload:
            media_content = upload.read()
        response = requests.put(
            GRAPH_API_ENDPOINT
            + f"/me/drive/items/root:/{DEST_FOLDER_NAME}/{file_name}:/content",
            headers=headers,
            data=media_content,
        )

        if response.status_code // 100 != 2:
            print(response)

        os.remove(src_path)  # delete tif after it's been sent off


def main():
    path_queue = queue.Queue()

    for _ in range(NUM_PRODUCERS):
        t.Thread(target=producer, args=(path_queue,)).start()

    for _ in range(NUM_CONSUMERS):
        t.Thread(target=consumer, args=(path_queue,)).start()


if __name__ == "__main__":
    main()
