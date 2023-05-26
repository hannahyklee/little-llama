import zstandard as zstd
import json
import io
import glob
import random

from paths import dir_val, dir_train
from typing import Union, List


# performs line formatting. Nothing really needs to be done except extract the relevant field from the json dict
def _read_line(line: str) -> Union[str, None]:

    line = line.strip()

    # might not read data properly(shouldn't happen with the TextIOWrapper thing, but just in case).
    try:
        data = json.loads(line)
    except ValueError:
        return None

    # if successful, data will be a dictionary, and we're only interested in the text.
    return data['text']


# reads num_per_chunk documents from a file to prevent loading all the file in memory
def _read_file_chunk(text_stream: io.TextIOWrapper, lines: List[str], num_per_chunk: int):

    full_chunk = False

    # add to list of lines
    for line in text_stream:

        line = _read_line(line)

        if line is not None:
            lines.append(line)

        if len(lines) == num_per_chunk:
            full_chunk = True
            break

    return lines, full_chunk


def read_data_folder(data_folder: str, num_per_chunk: int = 20000, randomize_files: bool = True):

    # get files in folder
    file_paths = glob.glob(data_folder + "*")
    file_paths = file_paths + file_paths

    # randomize file order
    if randomize_files:
        random.shuffle(file_paths)

    cur_file = 0    # index of the file being read
    lines = []      # stores the documents
    text_stream = None  # opened steam
    full_chunk = False  # flag indicating of the chunk is full

    # while there are still files to read
    while True:

        # if the chunk is not full, try to open a new file
        if not full_chunk:

            # close previous file which we completely read
            if text_stream:
                text_stream.close()

            # if there are no more files exit, but return if there is anything to return
            if cur_file == len(file_paths):
                if lines:
                    yield lines
                break

            # open file
            compressed_file = open(file_paths[cur_file], 'rb')
            # create decompressor
            decompressor = zstd.ZstdDecompressor()
            # Create a decompression byte stream
            stream_reader = decompressor.stream_reader(compressed_file)
            # Create a text stream
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

            cur_file += 1

        # try to read one full chunk or until the end of the current file
        lines, full_chunk = _read_file_chunk(text_stream, lines, num_per_chunk)

        # only return when chunks are full. Required since a file could terminate while the chunk is incomplete
        if full_chunk:
            yield lines
            lines = []


# some tests
if __name__ == "__main__":

    # for i, chunk in enumerate(read_data_folder(dir_val)):
    #     print(i, len(chunk))
    #     print(chunk[0][0:10])

    for i, chunk in enumerate(read_data_folder(dir_train)):
        print(i, len(chunk))
        print(chunk[0][0:10])
