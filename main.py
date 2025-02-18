import sys
import heapq
from collections import deque
from collections import defaultdict
# timer start
from atcoder.dsu import DSU
import bisect
import time
import random
start = time.time()
def main():
    # データの読み込み
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))  # グリッドサイズ
    M = int(next(it))  # 人物数
    K = int(next(it))  # 初期資金
    T = int(next(it))  # ターン数
    
    people = []
    points =set()
    for _ in range(M):
        r0 = int(next(it))
        c0 = int(next(it))
        r1 = int(next(it))
        c1 = int(next(it))
        points.add((r0,c0))
        points.add((r1,c1))
        people.append(((r0, c0), (r1, c1)))
    
    # 人をマンハッタン距離が大きい順にソート
    people.sort(key=lambda x: -sum(abs(x[0][i] - x[1][i]) for i in range(2)))

    COST_STATION = 5000  # 駅設置のコスト
    COST_RAIL = 100      # 線路設置のコスト

    # マンハッタン距離2以内の(dx, dy)リスト
    # moves = [
    #     (1, 0), (-1, 0), (0, 1), (0, -1),  # 距離1
    #     (2, 0), (-2, 0), (0, 2), (0, -2),  # 距離2 (縦横)
    #     (1, 1), (-1, -1), (1, -1), (-1, 1) # 距離2 (斜め)
    # ]
    moves = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),  # 距離1
        (2, 0), (-2, 0), (0, 2), (0, -2),  # 距離2 (縦横)
        (1, 1), (-1, -1), (1, -1), (-1, 1) # 距離2 (斜め)
    ]
    def manhattan(home, work):
        # マンハッタン距離を計算
        return abs(home[0] - work[0]) + abs(home[1] - work[1])
    def index(L):
        r,c=L
        return r*N+c
    
    def find_path(start, goal,connections):
        sr, sc = start
        gr, gc = goal

        # dijkstra で最短経路を探索
        if start == goal:
            return [start]
        
        # 距離の初期化
        dist = [[float('inf')] * N for _ in range(N)]
        dist[sr][sc] = 0
        prev = [[None] * N for _ in range(N)]  # 逆経路を保存
        
        # プライオリティキュー (距離, (row, col)) を使用
        pq = [(0, start)]  # (コスト, 座標)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while pq:
            cost, (r, c) = heapq.heappop(pq)
            
            # 目的地に到達した場合
            if (r, c) == goal:
                path = []
                while (r, c) != (sr, sc):
                    path.append((r, c))
                    r, c = prev[r][c]
                path.append((sr, sc))
                path.reverse()
                return path
            
            # 4方向の移動
            directions_t=directions.copy()
            if len(connections[(r,c)])>0:
                for ri,ci in connections[(r,c)]:
                    if ri!=r or ci!=c:
                        directions_t.append((ri-r,ci-c))
            for dr, dc in directions_t:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    # 移動コストを計算
                    if built[nr][nc] == 1:
                        next_cost = cost
                    elif built[nr][nc] == 2:
                        next_cost = cost + COST_STATION  # 線路はコスト5000
                    else:
                        next_cost = cost + COST_RAIL  # 更地はコスト100
                    
                    # 最短距離を更新
                    if next_cost < dist[nr][nc]:
                        dist[nr][nc] = next_cost
                        prev[nr][nc] = (r, c)
                        heapq.heappush(pq, (next_cost, (nr, nc)))
        
        return None  # 目標に到達できない場合

    def get_direction(p1, p2):
        # 2点間の方向を取得
        dr = p2[0] - p1[0]
        dc = p2[1] - p1[1]
        if dr == 1:
            return "down"
        if dr == -1:
            return "up"
        if dc == 1:
            return "right"
        if dc == -1:
            return "left"
        return None

    def generate_path_commands(path):
        # パスからコマンドを生成
        cmds = []
        L = len(path)
        r, c = path[0]

        if built[r][c] != 1:
            cmds.append(f"0 {r} {c}")  # 駅設置
        used[(r,c)]+=1 
        turning_map = {
            ("up", "right"): 6,
            ("right", "up"): 4,
            ("up", "left"): 3,
            ("left", "up"): 5,
            ("down", "right"): 5,
            ("right", "down"): 3,
            ("down", "left"): 4,
            ("left", "down"): 6,
        }
        for i in range(1, L - 1):
            prev = path[i - 1]
            cur = path[i]
            nxt = path[i + 1]
            d1 = get_direction(prev, cur)
            d2 = get_direction(cur, nxt)
            # debug
            # print(f"d1: {d1}, d2: {d2} prev: {prev} cur: {cur} nxt: {nxt}")
            # print(f"cur in connections[prev]: {cur in connections[prev]}")
            # print(f"nxt in connections[cur]: {nxt in connections[cur]}")

            if nxt in connections[cur] or cur in connections[prev]:
                continue
            if built[cur[0]][cur[1]] == 1:
                continue
            if d1 == d2:
                if d1 in ("left", "right"):
                    cmds.append(f"1 {cur[0]} {cur[1]}")  # 線路設置
                else:
                    cmds.append(f"2 {cur[0]} {cur[1]}")  # 縦方向の線路
            else:
                cmds.append(f"{turning_map[(d1, d2)]} {cur[0]} {cur[1]}")  # 方向転換のコマンド
        r, c = path[-1]
        if built[r][c] != 1:
            cmds.append(f"0 {r} {c}")  # 駅設置
        used[(r,c)]+=1
        return cmds

    def get_detour_commands(home, work,connections):
        path = find_path(home, work,connections)
        # print(f"path: {path}")
        if path is None:
            return []
        return generate_path_commands(path)

    # ここからメイン処理
    best_score = -float('inf')
    best_commands = []
    timeout=False
    for trial in range(1000):  # 10回繰り返し
        funds = K
        # built: 0: 更地, 1: 駅, 2: 線路
        used=defaultdict(int) 
        built = [[0] * N for _ in range(N)]
        output_commands = []
        connections=defaultdict(set)
        
        dsu=DSU(N*N)
        pep_t=people.copy()
        pep_t.sort(key=lambda x: sum(abs(x[0][i] - x[1][i]) for i in range(2)))


        
        pending = []
        current_person_income = None
        connected_incomes = 0
        home, work = None, None
        for turn in range(T):
            # check time
            if time.time() - start > 2.7:
                timeout=True
                break
            if best_score <funds+connected_incomes*(T-turn-1):
                output_commands_tmp=output_commands.copy()
                funds_tmp=funds+connected_incomes*(T-turn-1)
                while len(output_commands_tmp) < T:
                    output_commands_tmp.append("-1")
                best_score = funds_tmp
                best_commands = output_commands_tmp
                
            if not pending and len(pep_t)>0:

                def sort_key(x):
                    # sort by the value of the person
                    home, work = x
                    station_exist=0
                    for dx,dy in moves:
                        nx,ny=home[0]+dx,home[1]+dy
                        if 0 <= nx < N and 0 <= ny < N and used[(nx,ny)]<4 and built[nx][ny]==1:
                            station_exist+=1
                            home=(nx,ny)
                            break
                    for dx,dy in moves:
                        nx,ny=work[0]+dx,work[1]+dy
                        if 0 <= nx < N and 0 <= ny < N and used[(nx,ny)]<4 and built[nx][ny]==1:
                            station_exist+=1
                            work=(nx,ny)
                            break
                    dist = sum(abs(home[i] - work[i]) for i in range(2))-1
                    remaining_turns = T - turn - 1
                    cost = dist * 100 + 10000-station_exist*5000
                    if dsu.same(index(home),index(work)):
                        cost=0
                    value = cost - funds - connected_incomes * remaining_turns
                    rest_turns=0
                    if  connected_incomes == 0:
                        rest_turns=0 
                    else:
                        rest_turns=max((cost-funds)//connected_incomes,0)
                    # turn_needed=0
                    if value < 0:
                        n=(remaining_turns - max(dist,rest_turns))
                        # delta=0.9999
                        # return (dist+1) * (1-delta**n)/(1-delta)
                        return (dist+1) * n
                    else:
                        return -dist
              
                pep_t = sorted(pep_t, key=sort_key,reverse=True)
                idx=0
              
                # with some probability, index++
                if random.random() < 0.7 and len(pep_t)>1:
                    idx=random.choice(range(0,min(len(pep_t),10)))
                    # idx=random.choice(range(0,max(int(len(pep_t)//(800-turn)),5)))

                if idx<len(pep_t):
                    home, work = pep_t[idx]
                    current_person_income = manhattan(home, work)
                    del pep_t[idx]
                    # 近くにbuiltされた駅があるか探す
                    for dx,dy in moves:
                        nx,ny=home[0]+dx,home[1]+dy
                        if 0 <= nx < N and 0 <= ny < N and used[(nx,ny)]<4 and built[nx][ny]==1:
                            home=(nx,ny)
                            break
                    for dx,dy in moves:
                        nx,ny=work[0]+dx,work[1]+dy
                        if 0 <= nx < N and 0 <= ny < N and used[(nx,ny)]<4 and built[nx][ny]==1:
                            work=(nx,ny)
                            break
                    # 駅を置く場所をランダムに選ぶ
                    
                    if built[home[0]][home[1]]!=1:
                        max_point_cnt=0
                        max_home=None
                        for dx,dy in moves:
                            nx,ny=home[0]+dx,home[1]+dy
                            point_cnt=0
                            if 0 <= nx < N and 0 <= ny < N:
                                for dx2,dy2 in moves:
                                    nx2,ny2=nx+dx2,ny+dy2
                                    if (nx2,ny2) in points:
                                        point_cnt+=1
                            if point_cnt>max_point_cnt:
                                max_point_cnt=point_cnt
                                max_home=(nx,ny)
                        if max_home is not None:
                            home=max_home   
                                
                    if built[work[0]][work[1]]!=1:
                        max_point_cnt=0
                        max_work=None   
                        for dx,dy in moves:
                            nx,ny=work[0]+dx,work[1]+dy
                            point_cnt=0
                            if 0 <= nx < N and 0 <= ny < N:
                                for dx2,dy2 in moves:
                                    nx2,ny2=nx+dx2,ny+dy2
                                    if (nx2,ny2) in points:
                                        point_cnt+=1
                            if point_cnt>max_point_cnt:
                                max_point_cnt=point_cnt
                                max_work=(nx,ny)   
                        if max_work is not None:
                            work=max_work

                    if dsu.same(index(home),index(work)):
                        connected_incomes+=current_person_income
                        current_person_income=None
                        continue
                    if built[home[0]][home[1]]==1 and built[work[0]][work[1]]==1:
                        min_dist=manhattan(home,work)
                        home_tmp,work_tmp=home,work
                        for r1,c1 in connections[home]:
                            for r2,c2 in connections[work]:
                                dist=manhattan((r1,c1),(r2,c2))
                                if dist<min_dist:
                                    min_dist=dist
                                    home_tmp,work_tmp=(r1,c1),(r2,c2)
                        home,work=home_tmp,work_tmp
                    cmds = get_detour_commands(home, work,connections)
                else:
                    cmds = []
                if cmds:
                    pending = cmds
                    # print(f"current_person_income: {current_person_income}, connected_incomes: {connected_incomes}, turn: {turn}, funds: {funds}")
                else:
                    pending = []
                    current_person_income = 0

            if pending:
                cmd = pending[0]
                parts = cmd.split()
                cmd_type = parts[0]
                r = int(parts[1])
                c = int(parts[2])
                cost = 0
                if cmd_type=="0":  # 駅設置
                    cost = COST_STATION
                elif built[r][c] == 0:  # 更地を線路にする
                    cost = COST_RAIL
                elif built[r][c]==2:  # 線路を駅に変更
                    cmd = f"0 {r} {c}"
                    cmd_type = "0"
                    cost = COST_STATION
                
                if built[r][c]!=1 and funds >= cost:
                    if cmd_type == "0":
                        built[r][c] = 1  # 駅設置
                    else:
                        built[r][c] = 2  # 線路設置
                    funds -= cost
                    pending.pop(0)
                    output_commands.append(cmd)
                else:
                    output_commands.append("-1")
            else:
                output_commands.append("-1")
            
            if not pending and current_person_income is not None:
                connected_incomes += current_person_income
                dsu.merge(index(home),index(work))
                connections[home].add(home)
                connections[work].add(work)
                connections[home].add(work)
                connections[work].add(home)
                current_person_income = None
            funds += connected_incomes
        
        if funds > best_score:
            best_score = funds
            best_commands = output_commands
                # print(f"best_score: {best_score}")
                # print(f"best_commands: {best_commands}")
        if timeout:
            break
        
    sys.stdout.write("\n".join(best_commands[:T]))

if __name__ == "__main__":
    main()
