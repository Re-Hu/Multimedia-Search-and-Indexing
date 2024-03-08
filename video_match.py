import cv2
import numpy as np
import os
import sys
import json
import time
from sklearn.cluster import KMeans

# save the frame as capture.jpg at a given frame index, f_idx
def display_a_frame(name, f_idx):
    with open(name, 'rb') as file:
        video_data = file.read()
    
    num = 352*288*3
    start = f_idx * num
    end = (f_idx+1) * num
    video_frame = np.frombuffer(video_data[start:end], dtype=np.uint8).reshape(288, 352, 3)
    frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
    print(get_dominant_colors(frame))
    
    cv2.imwrite('capture2.jpg', frame)
    
# read RGB file and return a list of frames    
def read_video(name, query):
    width = 352
    height = 288
    video_name = name.split('/')[-1][:-4]
    with open(name, 'rb') as file:
        video_data = file.read()
    
    num = len(video_data)//(3*width*height)
    video_frames = np.frombuffer(video_data, dtype=np.uint8).reshape((num, height, width, 3))
    frames = []
    for f in video_frames:
        frames.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    
    # cv2.imwrite('capture.jpg', frames[100])
    # if not query: 
    #     path = './video_signatures/' + video_name
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     np.save(path+'/frames', frames)
    return frames

# input: name(video name), frames(a list of frames), query(boolean if input is query)

def create_signature(name, frames, query):
    name = name[:-4]
    signature = {'name': name}
    shots = []
    shots_idx = []
    colors = []
    color_theme = [0, 0, 0]
    frame_count = 0
    prev = None
    start = 0
    count = 0
    
    for f in frames:
        gray_tmp = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            frame_diff = cv2.absdiff(prev, gray_tmp)
            diff = frame_diff.mean()
            if diff > 10:
                shots_idx.append(start)
                shots.append(frame_count - start)
                frame_color = get_dominant_colors(frames[start]).astype(int)
                color_theme += frame_color
                colors.append(frame_color.tolist())
                start = frame_count
                count += 1
        prev = gray_tmp.copy()
        frame_count += 1
    
    shots_idx.append(start)
    shots.append(frame_count - start)
    frame_color = get_dominant_colors(frames[start]).astype(int)
    color_theme += frame_color
    colors.append(frame_color.tolist())
    count += 1
    signature['shots_idx'] = shots_idx
    signature['shots'] = shots
    signature['color_theme'] = (color_theme/count).astype(int).tolist()
    signature['colors'] = colors
    signature['shots_num'] = count
    
    
    # save the shot dictionary
    # if not query:
    #     if not os.path.exists('./video_signatures'):
    #         os.makedirs('./video_signatures')
    #     path = './video_signatures/' + name + '_signature.json'
    #     with open(path, "w") as outfile:
    #         json.dump(signature, outfile)
    # else:
    #     with open('query.json', "w") as outfile:
    #         json.dump(signature, outfile)
    return signature

def get_dominant_colors(frame):
    np.random.seed(42)
    pixels = cv2.resize(frame, (100, 100)).reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, n_init='auto')
    kmeans.fit(pixels)
    
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return dominant_color

def matching(signatures, query_signature):
    # start_time = time.time()
    diff = matching_color_theme(signatures, query_signature)
    # print(diff)
    start_idx = {}
    for k in diff:
        if query_signature['shots_num'] >= 3:
            ans = matching_shots(signatures[k]['shots'], query_signature['shots'])
            if ans:
                if len(ans) != 1:
                    return {}
                start_idx['name'] = k
                start_idx['frame#'] = signatures[k]['shots_idx'][ans[0]]-query_signature['shots_idx'][1]
                return start_idx
        elif query_signature['shots_num'] >= 2:
            ans = matching_colors(signatures[k], query_signature)
            if ans:
                if len(ans) != 1:
                    return {}
                start_idx['name'] = k
                start_idx['frame#'] = ans[0]-query_signature['shots'][0]
                return start_idx
            
    return start_idx

def matching_colors(signature, query_signature):
    ans = []
    j = 1
    for k in range(signature['shots_num']):
        color = signature['colors'][k]
        target = query_signature['colors'][j]
        if color == target:
            j += 1
            if j >= query_signature['shots_num']:
                ans.append(signature['shots_idx'][k-j+2])
                j = 1
        else:
            j = 1
        
    return ans

def matching_color_theme(signatures, query_signature):
    distances = {}
    for n in range(1, 21):
        video = 'video'+ str(n)
        d = euclidean_distance(signatures[video]['color_theme'], query_signature['color_theme'])
        distances[video] = d
        
    sorted_distances =  dict(sorted(distances.items(), key=lambda item: item[1]))
    return sorted_distances     
    
def matching_shots(shots, shots_q):
    if len(shots_q) < 3:
        return []
    tmp = shots_q[1:-1]
    matched_idx = []
    j = 0
    length_q = len(tmp)
    for i in range(len(shots)):
        if shots[i] == tmp[j]:
            j += 1
        else:
            j = 0
        if j >= length_q:
            matched_idx.append(i-j+1)
            j = 0
    
    return matched_idx

def euclidean_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))

def preprocess_video(path):
    files = os.listdir(path)
    for f in files:
        if f.endswith('.rgb'):
            print('loading {} ...'.format(f))
            frames = read_video('./Videos/RGB_Files/' + f, False)
            create_signature(f, frames, False)

def load_signatures():
    files = os.listdir('./video_signatures/')
    signatures = {}
    for sig in files:
        if sig.endswith('.json'):
            name = sig.strip('_signature.json')
            f = open('video_signatures/' + sig)
            signatures[name] = json.load(f)
    
    return signatures

# if __name__ == "__main__":
    
#     # preprocess_video('./Videos/RGB_Files/')
#     # preporcess_audio()
#     signatures = load_signatures()
    
#     query_video = sys.argv[1]
#     query_audio = sys.argv[2]
    
#     start = timeit.default_timer()
#     frames_q = read_video('./Queries/RGB_Files/' + query_video, True)
    
#     signature_q = create_signature(query_video, frames_q, True)
#     print(matching(signatures, signature_q))
    
#     rgb_time = timeit.default_timer()
#     print('rgb time consumed: ', rgb_time-start)
    
#     # audio_match('video1.wav', query_audio)
    
    
#     stop = timeit.default_timer()
#     # print('audio time consumed: ', stop-rgb_time)
    
#     print()
    
    
    
