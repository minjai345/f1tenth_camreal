"""Keep one message per topic every N messages (timestamps unchanged): a big camera bag -> a small one to upload.

Runs where ROS 2 is installed (the car). A 40 Hz bag with N=20 keeps one image per 0.5 s, all the labeling uses.
    python3 camreal/tools/decimate_bag.py data/bags/run_train data/bags/run_train_2hz 20
"""
import sys

import rosbag2_py

if len(sys.argv) != 4:
    sys.exit(__doc__)
src, dst, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=src, storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
writer = rosbag2_py.SequentialWriter()
writer.open(rosbag2_py.StorageOptions(uri=dst, storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
for topic in reader.get_all_topics_and_types():
    writer.create_topic(topic)
seen, kept = {}, {}
while reader.has_next():
    name, data, stamp = reader.read_next()
    seen[name] = seen.get(name, -1) + 1
    if seen[name] % step == 0:
        writer.write(name, data, stamp)
        kept[name] = kept.get(name, 0) + 1
del writer
print(f'{dst}: kept {kept} (every {step}th message per topic)')
