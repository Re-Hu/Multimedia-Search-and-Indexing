import cv2
import numpy as np
import os
import time
import librosa
from multiprocessing import Pool, Manager
import UIdisplay
from calcmfcc import preporcess_audio
import sys
import json
from sklearn.cluster import KMeans

def calculate_mfcc(audio_path, shared_mfcc_list):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
    shared_mfcc_list.append((mfcc, sr))

def calculate_mfcc2(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512)
    return mfcc, sr  # 同时返回 MFCC 和采样率

def find_match_in_mfcc(args):
    short_audio_mfcc, long_audio_mfcc, sr = args
    best_score = float('inf')
    best_start = None

    for start in range(long_audio_mfcc.shape[1] - short_audio_mfcc.shape[1]):
        window = long_audio_mfcc[:, start:start + short_audio_mfcc.shape[1]]
        score = np.mean(np.square(short_audio_mfcc - window))
        if score < best_score:
            best_score = score
            best_start = start

    return best_score, best_start * 512 / sr

def find_match_in_mfcc2(short_audio_mfcc, long_audio_mfcc):
    best_score = float('inf')
    best_start = None

    for start in range(long_audio_mfcc.shape[1] - short_audio_mfcc.shape[1]):
        window = long_audio_mfcc[:, start:start + short_audio_mfcc.shape[1]]
        score = np.mean(np.square(short_audio_mfcc - window))
        if score < best_score:
            best_score = score
            best_start = start

    return best_score, best_start

def process_file(args):
    mfcc_folder = 'Videos/MFCCs'
    file, short_audio_mfcc, sr = args
    if file.endswith('.npy'):
        mfcc_path = os.path.join(mfcc_folder, file)
        long_audio_mfcc = np.load(mfcc_path)
        score, match_start = find_match_in_mfcc((short_audio_mfcc, long_audio_mfcc, sr))
        return score, file, match_start
    
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

def create_signature(name, frames, query, d):
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
            if diff > d:
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
    if not query:
        if not os.path.exists('./video_signatures'):
            os.makedirs('./video_signatures')
        path = './video_signatures/' + name + '_signature.json'
        with open(path, "w") as outfile: 
            json.dump(signature, outfile)
    else:
        with open(f'query_{d}.json', "w") as outfile: 
            json.dump(signature, outfile)
    return signature

def get_dominant_colors(frame):
    np.random.seed(42)
    pixels = cv2.resize(frame, (100, 100)).reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, n_init='auto')
    kmeans.fit(pixels)
    
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return dominant_color

def matching(signatures, query_signature):
    diff = matching_color_theme(signatures, query_signature)
    # print(diff.keys())
    start_idx = {}
    files = []
    for k in diff:
        if query_signature['shots_num'] >= 3:
            ans = matching_shots(signatures[k]['shots'], query_signature['shots'])
            if ans:
                if len(ans) == 1:
                    start_idx['name'] = k
                    start_idx['frame#'] = signatures[k]['shots_idx'][ans[0]]-query_signature['shots_idx'][1]
                    return start_idx, None, []
                elif len(ans) > 1:
                    return {}, k, []
        elif query_signature['shots_num'] >= 2:
            ans = matching_colors(signatures[k], query_signature)
            if ans:
                # print(k)
                if len(ans) == 1:
                    start_idx['name'] = k
                    start_idx['frame#'] = ans[0]-query_signature['shots'][0]
                    return start_idx, None, []
                elif len(ans) > 1:
                    return {}, k, []
        else:
            ans = matching_colors2(signatures[k], query_signature)
            if ans:
                if contain(signatures[k]['shots'], 600):
                    files.append(k)
            
    if len(files) == 1:
        return {}, files[0], files
    elif len(files) > 1:
        return {}, None, files
    else:
        return {}, None, [f'video{x}' for x in range(1, 21)]

def matching2(signatures, query_signature):
    diff = matching_color_theme(signatures, query_signature)
    # print(diff.keys())
    files = []
    potentials = []
    for k in diff:
        if query_signature['shots_num'] > 3:
           ans = matching_shots(signatures[k]['shots'], query_signature['shots'])
           if ans:
                if len(ans) > 1:
                    potentials.append(k)
                else:
                    frame_n = signatures[k]['shots_idx'][ans[0]]-query_signature['shots_idx'][1]
                    files.append({'name': k, 'frame#': frame_n})
        elif query_signature['shots_num'] >= 2:
            ans = matching_colors(signatures[k], query_signature)
            if ans:
                if len(ans) > 1:
                    potentials.append(k)
                else:
                    frame_n = ans[0]-query_signature['shots'][0]
                    files.append({'name': k, 'frame#': frame_n})
        else:
            ans = matching_colors2(signatures[k], query_signature)
            if ans and contain(signatures[k]['shots'], 600):
                potentials.append(k)
    
    return files, potentials


def contain(shots, length):
    for shot in shots:
        if shot >= length:
            return True
    return False

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

def matching_colors2(signature, query_signature):
    ans = []
    j = 0
    for k in range(signature['shots_num']):
        color = signature['colors'][k]
        target = query_signature['colors'][j]
        if euclidean_distance(color, target) < 3:
            j += 1
            if j >= query_signature['shots_num']:
                ans.append(signature['shots_idx'][k-j+2])
                return ans
        else:
            j = 0
        
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

def load_signatures(d):
    files = os.listdir(f'./video_signatures_{d}/')
    signatures = {}
    for sig in files:
        if sig.endswith('.json'):
            name = sig.strip('_signature.json')
            f = open(f'video_signatures_{d}/' + sig)
            signatures[name] = json.load(f)
    
    return signatures

if __name__ == "__main__":
    # preprocess_video('./Videos/RGB_Files/')
    # preporcess_audio()
    print("start prepossing")
    query_video = sys.argv[1]
    query_audio = sys.argv[2]
    name = query_video.strip('.rgb')
    
    # # 加载预先计算好的文件
    mfcc_folder = 'Videos/MFCCs'
    # signatures = load_signatures()
    
    # # start timing
    # start_time = time.time()
    
    # # RGB match
    frames_q = read_video('./Queries/RGB_Files/' + query_video, True)
    # signature_q = create_signature(query_video, frames_q, True)
    # result = {}
    signatures = {}
    signature_q = {}
    for diff in [10, 15, 20, 25, 30]:
        signatures[diff] = load_signatures(diff)
        signature_q[diff] = create_signature(query_video, frames_q, True, diff)
    
    print("Done preprocessing \nstart matching")
    
    potentials = set()
    result = {}
    start_time = time.time()
    for diff in [10, 15, 20, 25, 30]:
        res, files = matching2(signatures[diff], signature_q[diff])
        if len(res) == 1:
            result = res[0]
            t = time.time() - start_time
            break
        if files:
            potentials.update(set(files))
    
    potentials = list(potentials)
   
    if result:        
        match_video = result['name']
        match_frame = result['frame#']
    elif len(potentials) == 1:
        ans = potentials[0]
        manager = Manager()
        shared_mfcc_list = manager.list()
        short_audio_mfcc, sr = calculate_mfcc2('Queries/Audios/' + query_audio)
        mfcc_folder = 'Videos/MFCCs'
        mfcc_path = os.path.join(mfcc_folder, ans+'.wav' + ".npy")
        long_audio_mfcc = np.load(mfcc_path)

        start_time = time.time()
        score, match_start = find_match_in_mfcc2(short_audio_mfcc, long_audio_mfcc)
        best_match_time = match_start * 512 / sr
        match_frame = round(best_match_time * 30)
        match_video = ans
        end_time = time.time()
        t = end_time - start_time
    else:
        # Audio Match
        manager = Manager()
        shared_mfcc_list = manager.list()

        # Calculate MFCC for the short audio file
        calculate_mfcc(f'Queries/Audios/{query_audio}', shared_mfcc_list)

        # Get the calculated MFCC and sample rate
        short_audio_mfcc, sr = shared_mfcc_list[0]
        potentials = [file+'.wav.npy' for file in potentials]

        # 开始计时
        start_time = time.time()

        best_score = float('inf')
        best_match_time = None
        match_video = None

        # Process files in parallel and collect results
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(process_file, [(file, short_audio_mfcc, sr) for file in potentials])
        results = [result for result in results if result is not None]
        # print("results: ", results)

        # Find the best match among results
        for score, file, match_start in results:
            if score < best_score:
                best_score = score
                best_match_time = match_start
                match_video = file.rstrip('.wav.npy')

        frame_rate = 30  # 每秒帧数
        match_frame = round(best_match_time * frame_rate)  # 使用四舍五入转换匹配时间为帧
        end_time = time.time()
        t = end_time - start_time
        
    # # end timing and print result
        
    print(f"Find the best match of query {name} in {match_video} at frame #{match_frame}.")
    print(f"time consumed: {t} seconds \n\nstart UI displaying")


    # video_file = 'Queries/RGB_Files/video4_1.rgb'
    # video_file = 'Videos/RGB_Files/{}'.format(match_video + '.rgb')
    video_file = 'Videos/{}'.format(match_video+'.mp4')
    audio_file = 'Videos/Audios/{}'.format(match_video+'.wav')
    output_video_file = 'output_video.mp4'
    # audio_file = "Queries/Audios/video4_1.wav"
    # frame_i = 200

    video_player = UIdisplay.VideoPlayer(match_frame, output_video_file, video_file, audio_file)
    video_player.run()


