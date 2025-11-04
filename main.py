from utils.video_utils import read_video, save_video
import os
os.makedirs("output_videos", exist_ok=True)
from trackers.tracker import Tracker
import cv2

def main():
    #read video
    video_frames = read_video('input_video_data/soccer_video_4.mp4')
    
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    #saving and cropping image of a player
    for track_id,player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]
        cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        cv2.imwrite(f"ouput_videos/cropped_image.jpg", cropped_image)
        break
    
    #draw ouput
    output_video_frames = tracker.draw_anotations(video_frames, tracks)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == "__main__":
    main()