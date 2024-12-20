from presence import PresenceTracker

detections = [
    (0, 0.9)
]

frame_time = 33_333

tracker = PresenceTracker()

for i in range(1, 10):
    t = frame_time*i
    print("Updating presence tracker with batch of detections at time", t)
    tracker.update_batch(t, detections, 0.5)

all_tracks = tracker.list_presence_timestamps(0, 0, frame_time * 1000)
print("All tracks:", all_tracks)