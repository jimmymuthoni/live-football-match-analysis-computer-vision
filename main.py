from utils.video_utils import read_video, save_video
import os
os.makedirs("output_videos", exist_ok=True)
from trackers.tracker import Tracker
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from team_assigner.team_assigner import TeamAssigner
import cv2
import numpy as np

def main():
    #read video
    video_frames = read_video('input_video_data/soccer_video_4.mp4')
    
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    #interpolate ball position
    tracks['ball'] = tracker.interpolate_ball_position(tracks["ball"])
    
    #assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id,track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'],player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    #assign ball aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num,player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])

        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)



    #draw ouput
    output_video_frames = tracker.draw_anotations(video_frames, tracks, team_ball_control)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == "__main__":
    main()