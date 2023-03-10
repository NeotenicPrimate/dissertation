import numpy as np
import pyautogui as ag
import os
import time
import sys

num_docs = int(sys.argv[1])

start = np.arange(1, num_docs+1, 500)
end = np.arange(500, num_docs+1, 500)

doc_ranges = list(zip(start, end))

if not end[-1] == num_docs:
    doc_ranges.append((end[-1]+1, num_docs))

time.sleep(2)

current_len_downloads = 0
for (start_record, end_record) in doc_ranges:

    export_button_1 = (758, 481)
    ag.click(export_button_1)
    time.sleep(0.5)

    # tab_delimited_file_button = (765, 694)
    tab_delimited_file_button = (779, 725)
    # tab_delimited_file_button = (817, 337)
    ag.click(tab_delimited_file_button)
    time.sleep(0.5)

    records_from_button = (553, 586)
    # records_from_button = (554, 573)
    ag.click(records_from_button)
    time.sleep(0.5)

    first_number = (681, 598)
    # first_number = (682, 587)
    ag.click(first_number)
    with ag.hold('command'):
        ag.press('a')
    ag.write(str(start_record))
    time.sleep(0.5)

    second_number = (768, 598)
    # second_number = (766, 586)
    ag.click(second_number)
    with ag.hold('command'):
        ag.press('a')
    ag.write(str(end_record))
    time.sleep(0.5)

    record_content = (908, 720)
    ag.click(record_content)
    
    time.sleep(0.5)

    full_record_and_cited_references = (747, 822)
    ag.click(full_record_and_cited_references)
    time.sleep(0.5)

    export_button_2 = (568, 772)
    ag.click(export_button_2)
    
    while True:
        
        download_files = os.listdir('/Users/vladimirborel/Downloads/')
        filtered_download_files = list(filter(lambda s: s.startswith('savedrecs'), download_files))
        len_filtered_download_files = len(filtered_download_files)

        if len_filtered_download_files != current_len_downloads:
            current_len_downloads += 1
            break
    
    time.sleep(0.5)

print(f'{np.ceil(num_docs/500)} files downloaded')
