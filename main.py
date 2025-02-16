import sys
import heapq
from collections import deque
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

    def find_path(start, goal):
        sr, sc = start
        gr, gc = goal
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
            for dr, dc in directions:
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

    def get_detour_commands(home, work):
        path = find_path(home, work)
        if path is None:
            return []
        return generate_path_commands(path)

    best_score = -float('inf')
    best_commands = []

    for trial in range(200):  # 10回繰り返し
        funds = K
        # built: 0: 更地, 1: 駅, 2: 線路
        built = [[0] * N for _ in range(N)]
        output_commands = []
        # current_person_index = 0
        # while (manhattan(people[current_person_index][0], people[current_person_index][1]))*100+10000 > funds:
        #     current_person_index += 1
        # manhattan でソートして、funds以下で最大の人を選ぶ
        pep_t=people.copy()
        pep_t.sort(key=lambda x: sum(abs(x[0][i] - x[1][i]) for i in range(2)))

        
        # delete pep_init
        
        pending = []
        current_person_income = None
        connected_incomes = 0
        home, work = None, None
        for turn in range(T):
            # check time
            if time.time() - start > 2.7:
                trial=800
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
            if not pending and len(pep_t)>0:
                # idx=bisect.bisect_right([sum(abs(x[0][i] - x[1][i]) for i in range(2))*100+10000 for x in pep_t],funds+connected_incomes*(T-turn-1))
                def sequence(N):
                    return 1 +  ((N ** 0.5 - 1) / (2000 ** 0.5 - 1))/2

                def sort_key(x):
                    dist = sum(abs(x[0][i] - x[1][i]) for i in range(2))
                    remaining_turns = T - turn - 1
                    add_cost = sequence(M-len(pep_t))
                    # add_cost =1
                    value = dist*add_cost * 100 + 10000 - funds - connected_incomes * remaining_turns
                    turn_needed=0
                    if remaining_turns == 0:
                        turn_needed=0 
                    else:
                        turn_needed=(dist*add_cost*100+10000-funds)//remaining_turns
                    # turn_needed=0
                    if value < 0:
                        if turn_needed<dist:
                            return dist * (remaining_turns - dist*add_cost)
                        else:
                            return dist*(remaining_turns-turn_needed)

                    else:
                        return -1e9+dist

                pep_t = sorted(pep_t, key=sort_key,reverse=True)
                idx=0
                # with some probability, index++
                if random.random() <.7 and len(pep_t)>1:
                    idx=random.choice(range(0,min(len(pep_t),10)))

                if idx<len(pep_t):
                    home, work = pep_t[idx]
                    del pep_t[idx]
                    cmds = get_detour_commands(home, work)
                else:
                    cmds = []
                if cmds:
                    pending = cmds
                    current_person_income = manhattan(home, work)
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
                current_person_income = None
            funds += connected_incomes
        
        if funds > best_score:
            best_score = funds
            best_commands = output_commands
                # print(f"best_score: {best_score}")
                # print(f"best_commands: {best_commands}")

    sys.stdout.write("\n".join(best_commands[:T]))

if __name__ == "__main__":
    main()
