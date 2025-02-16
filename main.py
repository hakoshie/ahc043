import sys
import heapq
from collections import deque
from collections import defaultdict
# timer start
import bisect
import time
import random
start = time.time()
def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))  # グリッドサイズ
    M = int(next(it))  # 人物数
    K = int(next(it))  # 初期資金
    T = int(next(it))  # ターン数
    V = N*N  # vertex数
    people = []
    for _ in range(M):
        r0 = int(next(it))
        c0 = int(next(it))
        r1 = int(next(it))
        c1 = int(next(it))
        people.append(((r0, c0), (r1, c1)))
    
    # 人をマンハッタン距離が大きい順にソート
    people.sort(key=lambda x: -sum(abs(x[0][i] - x[1][i]) for i in range(2)))

    COST_STATION = 5000  # 駅設置のコスト
    COST_RAIL = 100      # 線路設置のコスト

    def manhattan(home, work):
        return abs(home[0] - work[0]) + abs(home[1] - work[1])


    def find_path(start, goal, connections,ret_dist=False):
        sr, sc = start
        gr, gc = goal
        if start == goal:
            if ret_dist:
                return 0
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
                if ret_dist:
                    return dist[gr][gc]
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
        cmds = []
        L = len(path)
        r, c = path[0]

        if built[r][c] != 1:
            cmds.append(f"0 {r} {c}")  # 駅設置
        for i in range(1, L - 1):
            prev = path[i - 1]
            cur = path[i]
            nxt = path[i + 1]
            d1 = get_direction(prev, cur)
            d2 = get_direction(cur, nxt)
            if built[cur[0]][cur[1]] == 1:
                continue
            if d1 == d2:
                if d1 in ("left", "right"):
                    cmds.append(f"1 {cur[0]} {cur[1]}")  # 線路設置
                else:
                    cmds.append(f"2 {cur[0]} {cur[1]}")  # 縦方向の線路
            else:
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
                cmds.append(f"{turning_map[(d1, d2)]} {cur[0]} {cur[1]}")  # 方向転換のコマンド
        r, c = path[-1]
        if built[r][c] != 1:
            cmds.append(f"0 {r} {c}")  # 駅設置
        return cmds

    def get_detour_commands(home, work,connections):
        path = find_path(home, work,connections)
        if path is None:
            return []
        return generate_path_commands(path)
    def index(L):
        x,y=L
        return x * N + y
    
    best_score = -float('inf')
    best_commands = []
    exit_loop=False
    for trial in range(200):  # 10回繰り返し
        if time.time() - start > 2.7:
            # print("time out trial:",trial)
            # turn=1000
            trial=800
            break
        funds = K
        # built: 0: 更地, 1: 駅, 2: 線路
        built = [[0] * N for _ in range(N)]
        output_commands = []
        connections=defaultdict(list)
        dist=[[float('inf')]*V for _ in range(V)]

        # current_person_index = 0
        # while (manhattan(people[current_person_index][0], people[current_person_index][1]))*100+10000 > funds:
        #     current_person_index += 1
        # manhattan でソートして、funds以下で最大の人を選ぶ
        people_t=people.copy()
        for pep in people_t:
            home,work=pep
            dist[index(home)][index(work)]=dist[index(work)][index(home)]=manhattan(home,work)


        
        # delete pep_init
        
        pending = []
        current_person_income = None
        connected_incomes = 0
        home, work = None, None
        for turn in range(T):
            # check time
            if time.time() - start > 2.7:
                exit_loop=True
                # print("time out","trial:",trial, "turn:",turn)
                
                break
            if best_score <funds+connected_incomes*(T-turn-1):
                output_commands_tmp=output_commands.copy()
                funds_tmp=funds+connected_incomes*(T-turn-1)
                while len(output_commands_tmp) < T:
                    output_commands_tmp.append("-1")
                best_score = funds_tmp
                best_commands = output_commands_tmp
                
            # if turn >= T - rest:
            #     # print(f"before rest: {rest}, turn: {turn}, funds: {funds},connected_incomes: {connected_incomes}")
            #     while len(output_commands) < T:
            #         output_commands.append("-1")
            #         funds += connected_incomes
            #     # print(f"after  rest: {rest}, turn: {turn}, funds: {funds},connected_incomes: {connected_incomes}")
            #     break
            if not pending and len(people_t)>0:
                # idx=bisect.bisect_right([sum(abs(x[0][i] - x[1][i]) for i in range(2))*100+10000 for x in pep_t],funds+connected_incomes*(T-turn-1))

                def sort_key(x):
                    manhat_d = sum(abs(x[0][i] - x[1][i]) for i in range(2))
                    distance=dist[index(x[0])][index(x[1])]

                    remaining_turns = T - turn - 1 

                    reachable= funds+connected_incomes*(T-turn-1)-distance*100-10000
                    turn_needed=max((distance*100+10000-funds)//100,0)
                    # turn_needed=0
                    if reachable > 0:
                        return (remaining_turns-max(turn_needed,manhat_d))*manhat_d-distance*100-10000
                    else:
                        return -1e9+manhat_d
                # dijkstra for each person
     
                            
                people_t = sorted(people_t, key=sort_key,reverse=True)
                # if turn==0:
                #     for pep in people_t:
                #         home,work=pep
                        # print(f"home: {home}, work: {work}, dist: {dist[index(home)][index(work)]}")
                idx=0
                # with some probability, index++
                # if connected_incomes>0:
                if len(people_t)>1:
                    idx=random.choice(range(0,min(len(people_t),2)))
                # if random.random() < 0.5 and len(people_t)>1:
                #     idx=random.choice(range(0,min(len(people_t),10)))

                if idx<len(people_t):
                    home, work = people_t[idx]
                    del people_t[idx]
                    # print(f"home: {home}, work: {work}","get_detour_commands start","manhattan",manhattan(home,work),len(people_t),len(people))
                    cmds = get_detour_commands(home, work,connections)
                else:
                    cmds = []
                if cmds:
                    pending = cmds
                    current_person_income = manhattan(home, work)
                    # print(f"current_person_income: {current_person_income}, connected_incomes: {connected_incomes}, turn: {turn}, funds: {funds}")
                else:
                    pending = []
                    current_person_income = 0
            # print("pending size",len(pending))
            # if len(pending)==1:
            #     print("pending",pending)
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
                # if len(pending)==1:
                # print("turn",turn,"cmd",cmd,"cost",cost,"funds",funds,"built",built[r][c],"connected_incomes",connected_incomes)
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
                connections[home].append(work)
                connections[work].append(home)
                # print("connected",home,work)
                # dist[index(home[0],home[1])][index(work[0],work[1])]=0
                # dist[index(work[0],work[1])][index(home[0],home[1])]=0
                for pep in people_t:
                    home_t,work_t=pep
                    # check time
                    if time.time() - start > 2.7:
                        exit_loop=True
                        # print("time out","trial:",trial, "turn:",turn)
                        break
                    dist[index(home_t)][index(work_t)]=dist[index(work_t)][index(home_t)]=find_path(home_t,work_t,connections,ret_dist=True)
                # home, work = None, None
                current_person_income = None
            funds += connected_incomes
        
        if funds > best_score:
            best_score = funds
            best_commands = output_commands
                # print(f"best_score: {best_score}")
                # print(f"best_commands: {best_commands}")
        if exit_loop:
            break
    sys.stdout.write("\n".join(best_commands[:T]))

if __name__ == "__main__":
    main()
