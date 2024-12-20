from presence import PresenceTracker

# Create a new tracker (default 30 FPS)
tracker = PresenceTracker()

# Track some detections
detections = [(0, 0.9)]  # (class_id, confidence)
frame_time = 33_333  # microseconds (about 30 FPS)

# Update multiple frames
for i in range(1, 10):
    t = frame_time * i

    # only add detections every other frame
    if i % 2 == 0:
        tracker.update_batch(t, detections, confidence_threshold=0.5)
    else:
        tracker.update_batch(t, [], confidence_threshold=0.5)

# Query all presence timestamps
timestamps = tracker.list_presence_timestamps(
    class_id=0, start_time=0, end_time=frame_time * 1000
)

# convert timestamps to back to frame numbers
frames = [t // frame_time for t in timestamps]

print("Presence timestamps:", frames)
# Presence timestamps: [2, 4, 6, 8]
