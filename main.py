import sys
import heapq
from collections import deque, defaultdict
from atcoder.dsu import DSU
import bisect
import time
import random
import math

start = time.time()


def read_input():
    """標準入力からパラメータ、人物情報、初期ポイントを読み込む"""
    data = sys.stdin.read().strip().split()
    if not data:
        return None, None, None, None, None
    it = iter(data)
    N = int(next(it))  # グリッドサイズ
    M = int(next(it))  # 人物数
    K = int(next(it))  # 初期資金
    T = int(next(it))  # ターン数

    people = []
    points_ini = set()
    for _ in range(M):
        r0 = int(next(it))
        c0 = int(next(it))
        r1 = int(next(it))
        c1 = int(next(it))
        points_ini.add((r0, c0))
        points_ini.add((r1, c1))
        people.append(((r0, c0), (r1, c1)))
    
    return N, M, K, T, people, points_ini


def manhattan(home, work):
    """マンハッタン距離を計算する"""
    return abs(home[0] - work[0]) + abs(home[1] - work[1])


def index(L, N):
    """(r, c) を1次元のインデックスに変換する"""
    r, c = L
    return r * N + c


def index_inv(i, N):
    """1次元インデックスを (r, c) に戻す"""
    return i // N, i % N


def vacant_station(r, c, built, N):
    """(r, c) の隣接区画に、更地（built == 0）があるかチェックする"""
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    for i in range(4):
        nx, ny = r + dx[i], c + dy[i]
        if 0 <= nx < N and 0 <= ny < N and built[nx][ny] == 0:
            return True
    return False


def find_path(start, goal, connections, dsu, N, built, COST_STATION, COST_RAIL, turn,stations, ret_dist=False):
    """
    始点から終点までの最短経路を、コスト付きダイクストラ法により探索する。
    built の値に応じて移動コストを設定。
    """
    sr, sc = start
    gr, gc = goal

    if start == goal:
        return [start]
    if dsu.same(index(start, N), index(goal, N)):
        return None
    
    dist = [[float('inf')] * N for _ in range(N)]
    dist[sr][sc] = 0
    prev = [[None] * N for _ in range(N)]
    
    pq = [(0, start)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while pq:
        cost, (r, c) = heapq.heappop(pq)
        
        if (r, c) == goal:
            if ret_dist:
                return dist[r][c]
            # print(f"manhat: {manhattan(start,goal)} actual:{dist[r][c]/100} ratio:{dist[r][c]/100/manhattan(start,goal):.3f} turn:{turn} stations:{len(stations)}")    
            # 経路を復元する
            path = []
            while (r, c) != (sr, sc):
                path.append((r, c))
                r, c = prev[r][c]
            path.append((sr, sc))
            path.reverse()
            return path
        
        # 通常の4方向に加え、接続情報からも方向を追加
        directions_t = directions.copy()
        leader_id = dsu.leader(index((r, c), N))
        for id in connections[leader_id]:
            ri, ci = index_inv(id, N)
            if (ri, ci) != (r, c):
                directions_t.append((ri - r, ci - c))
        for dr, dc in directions_t:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N:
                if built[nr][nc] == 1:
                    next_cost = cost
                elif built[nr][nc] == 2:
                    next_cost = cost + COST_STATION
                else:
                    next_cost = cost + COST_RAIL
                
                if next_cost < dist[nr][nc]:
                    dist[nr][nc] = next_cost
                    prev[nr][nc] = (r, c)
                    heapq.heappush(pq, (next_cost, (nr, nc)))
    
    return None


def get_direction(p1, p2):
    """p1 から p2 への方向（up/down/left/right）を返す"""
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


def generate_path_commands(path, connections, dsu, N, built, COST_STATION):
    """
    探索したパスから、各セルでの建設コマンドを生成する。
    駅配置（"0"）や線路配置（"1", "2", "3"～"6"）の指示を返す。
    """
    cmds = []
    L = len(path)
    start, goal = path[0], path[-1]
    r, c = path[0]

    if built[r][c] != 1:
        cmds.append(f"0 {r} {c}")
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
        prev_pt = path[i - 1]
        cur_pt = path[i]
        next_pt = path[i + 1]
        d1 = get_direction(prev_pt, cur_pt)
        d2 = get_direction(cur_pt, next_pt)
        if built[cur_pt[0]][cur_pt[1]] == 1:
            dsu.merge(index(start, N), index(cur_pt, N))
            continue
        if built[cur_pt[0]][cur_pt[1]]==2:
            cmds.append(f"0 {cur_pt[0]} {cur_pt[1]}")
            dsu.merge(index(start, N), index(cur_pt, N))
            continue
        if d1 == d2:
            if d1 in ("left", "right"):
                cmds.append(f"1 {cur_pt[0]} {cur_pt[1]}")
            else:
                cmds.append(f"2 {cur_pt[0]} {cur_pt[1]}")
        else:
            cmds.append(f"{turning_map[(d1, d2)]} {cur_pt[0]} {cur_pt[1]}")
    r, c = path[-1]
    if built[r][c] != 1:
        cmds.append(f"0 {r} {c}")
    return cmds


def get_detour_commands(home, work, connections, dsu, N, built, COST_STATION, COST_RAIL,turn,stations):
    """
    指定された home から work への迂回経路のコマンド列を取得する
    """
    path = find_path(home, work, connections, dsu, N, built, COST_STATION, COST_RAIL,turn,stations)
    if path is None:
        return []
    return generate_path_commands(path, connections, dsu, N, built, COST_STATION)


def pep2points(pep):
    """人物リストから家・職場の位置集合を抽出する"""
    points = set()
    for p in pep:
        points.add(p[0])
        points.add(p[1])
    return points


def get_group_members(dsu, x, stations, N):
    """DSU のグループに属する駅リストを返す"""
    leader_x = dsu.leader(x)
    return [station for station in stations if dsu.leader(index(station, N)) == leader_x]


def find_nearest_point(start, candidates):
    """start からのマンハッタン距離が最小となる候補点を返す"""
    min_dist = 1e9
    nearest = start
    for r, c in candidates:
        dist = manhattan(start, (r, c))
        if dist < min_dist:
            min_dist = dist
            nearest = (r, c)
    return nearest


def main():
    """メインのシミュレーションループ"""
    N, M, K, T, people, points_ini = read_input()
    if N is None:
        return

    # 人物は家と職場の距離が大きい順にソート
    people.sort(key=lambda x: -sum(abs(x[0][i] - x[1][i]) for i in range(2)))

    COST_STATION = 5000
    COST_RAIL = 100

    moves = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]

    best_score = -float('inf')
    best_commands = []
    timeout = False

    # 1000回のシミュレーション試行
    for _ in range(1000):
        funds = K
        built = [[0] * N for _ in range(N)]
        output_commands = []
        connections = defaultdict(set)
        stations = set()
        dsu = DSU(N * N)
        for group in dsu.groups():
            connections[dsu.leader(group[0])] = group

        pep_t = people.copy()
        pep_t.sort(key=lambda x: sum(abs(x[0][i] - x[1][i]) for i in range(2)))
        
        pending = []
        current_person_income = None
        connected_incomes = 0
        home, work = None, None

        # 各ターンのシミュレーション
        for turn in range(T):
            if time.time() - start > 2.7:
                timeout = True
                break

            # 現在の資金と将来の収入で最良のスコアなら更新
            if best_score < funds + connected_incomes * (T - turn - 1):
                output_commands_tmp = output_commands.copy()
                funds_tmp = funds + connected_incomes * (T - turn - 1)
                while len(output_commands_tmp) < T:
                    output_commands_tmp.append("-1")
                best_score = funds_tmp
                best_commands = output_commands_tmp

            # 未処理のコマンドがなく、かつまだ人物が残っていれば、次の人物を選ぶ
            if not pending and len(pep_t) > 0:
                def sort_key(x):
                    home_x, work_x = x
                    dist_m = sum(abs(home_x[i] - work_x[i]) for i in range(2)) - 1
                    station_exist = 0
                    # 付近の駅を探索
                    home_stations = []
                    work_stations = []
                    for dx, dy in moves:
                        nx, ny = home_x[0] + dx, home_x[1] + dy
                        if 0 <= nx < N and 0 <= ny < N and built[nx][ny] == 1:
                            home_stations.append((nx, ny))
                    for dx, dy in moves:
                        nx, ny = work_x[0] + dx, work_x[1] + dy
                        if 0 <= nx < N and 0 <= ny < N and built[nx][ny] == 1:
                            work_stations.append((nx, ny))
                    station_exist += 1 if home_stations else 0
                    station_exist += 1 if work_stations else 0
                    home_x=random.choice(home_stations) if home_stations else home_x
                    work_x=random.choice(work_stations) if work_stations else work_x
                    # dist=find_path(home_x,work_x,connections,dsu,N,built,COST_STATION,COST_RAIL,ret_dist=True)
                    dist= dist_m
                    cost = dist * 100 + 10000 - station_exist * 5000
                    for home_station in home_stations:
                        exit_flag = False
                        for work_station in work_stations:
                            dist_t = sum(abs(home_station[i] - work_station[i]) for i in range(2)) - 1
                            # dist_t=find_path(home_station,work_station,connections,dsu,N,built,COST_STATION,COST_RAIL,ret_dist=True)
                            cost = min(cost, dist_t * 100 + 10000 - station_exist * 5000)
                            if dsu.same(index(home_station, N), index(work_station, N)):
                                cost = 0
                                exit_flag = True
                                break
                        if exit_flag:
                            break
                    remaining_turns = T - turn - 1
                    reachable =  funds + connected_incomes * remaining_turns -cost
                    rest_turns = 0 if connected_incomes == 0 else max(math.ceil((cost - funds) / connected_incomes), 0)
                    if reachable > 0:
                        n = (remaining_turns - max(dist_m, rest_turns))
                        return (dist_m + 1) * n / (cost + 1)
                    else:
                        return -(dist_m + 1)

                delete_ids = []
                for i, (home_x, work_x) in enumerate(pep_t):
                    home_stations = []
                    work_stations = []
                    for dx, dy in moves:
                        nx, ny = home_x[0] + dx, home_x[1] + dy
                        if 0 <= nx < N and 0 <= ny < N and built[nx][ny] == 1:
                            home_stations.append((nx, ny))
                    for dx, dy in moves:
                        nx, ny = work_x[0] + dx, work_x[1] + dy
                        if 0 <= nx < N and 0 <= ny < N and built[nx][ny] == 1:
                            work_stations.append((nx, ny))
                    exit_flag = False
                    for home_station in home_stations:
                        for work_station in work_stations:
                            if dsu.same(index(home_station, N), index(work_station, N)):
                                delete_ids.append(i)
                                exit_flag = True
                                break
                        if exit_flag:
                            break
                for i in delete_ids:
                    connected_incomes += sum(abs(pep_t[i][0][j] - pep_t[i][1][j]) for j in range(2))
                pep_t = [pep_t[i] for i in range(len(pep_t)) if i not in delete_ids]

                pep_t = sorted(pep_t, key=sort_key, reverse=True)
                idx = 0
                if random.random() < 0.7 and len(pep_t) > 1:
                    idx = random.choice(range(0, min(len(pep_t), 10)))

                if idx < len(pep_t):
                    home, work = pep_t[idx]
                    current_person_income = manhattan(home, work)
                    del pep_t[idx]
                    homes=[]
                    works=[]
                    for dx, dy in moves:
                        nx, ny = home[0] + dx, home[1] + dy
                        if 0 <= nx < N and 0 <= ny < N and built[nx][ny] == 1 and vacant_station(nx, ny, built, N):
                            homes.append((nx, ny))
                
                    for dx, dy in moves:
                        nx, ny = work[0] + dx, work[1] + dy
                        if 0 <= nx < N and 0 <= ny < N and built[nx][ny] == 1 and vacant_station(nx, ny, built, N):
                            works.append((nx, ny))
                            
                    if homes and works:
                        min_dist = 1e9
                        for home_station in homes:
                            for work_station in works:
                                dist = sum(abs(home_station[i] - work_station[i]) for i in range(2))
                                if dist < min_dist:
                                    min_dist = dist
                                    home = home_station
                                    work = work_station
                    elif homes:
                        home=find_nearest_point(work, homes) if homes else home
                    elif works:
                        work=find_nearest_point(home, works) if works else work
                    points = pep2points(pep_t)
                    if built[home[0]][home[1]] != 1:
                        max_point_cnt = 0
                        max_home = None
                        for dx, dy in moves:
                            nx, ny = home[0] + dx, home[1] + dy
                            point_cnt = 0
                            if 0 <= nx < N and 0 <= ny < N:
                                for dx2, dy2 in moves:
                                    nx2, ny2 = nx + dx2, ny + dy2
                                    if (nx2, ny2) in points_ini:
                                        point_cnt += 1
                                        if (nx2, ny2) in points:
                                            point_cnt += 1
                                    
                            if point_cnt > max_point_cnt:
                                max_point_cnt = point_cnt
                                max_home = (nx, ny)
                        if max_home is not None:
                            home = max_home
                    if built[work[0]][work[1]] != 1:
                        max_point_cnt = 0
                        max_work = None
                        for dx, dy in moves:
                            nx, ny = work[0] + dx, work[1] + dy
                            point_cnt = 0
                            if 0 <= nx < N and 0 <= ny < N:
                                for dx2, dy2 in moves:
                                    nx2, ny2 = nx + dx2, ny + dy2
                                    if (nx2, ny2) in points_ini:
                                        point_cnt += 1
                                        if (nx2, ny2) in points:
                                            point_cnt += 1
                                    
                            if point_cnt > max_point_cnt:
                                max_point_cnt = point_cnt
                                max_work = (nx, ny)
                        if max_work is not None:
                            work = max_work
                    
                    if built[home[0]][home[1]] == 1 or built[work[0]][work[1]] == 1:
                        if built[home[0]][home[1]] == 1 and built[work[0]][work[1]] == 1:
                            home_candidates = get_group_members(dsu, index(home, N), stations, N)
                            work_candidates = get_group_members(dsu, index(work, N), stations, N)
                            _, home, work = min(
                                ((manhattan((r1, c1), (r2, c2)), (r1, c1), (r2, c2)))
                                for r1, c1 in home_candidates
                                for r2, c2 in work_candidates
                            )
                        else:
                            if built[work[0]][work[1]] == 1:
                                home, work = work, home
                            home = find_nearest_point(work, get_group_members(dsu, index(home, N), stations, N))
                    
                    cmds = get_detour_commands(home, work, connections, dsu, N, built, COST_STATION, COST_RAIL,turn,stations)
                else:
                    cmds = []
                if cmds:
                    pending = cmds
                else:
                    pending = []
                    current_person_income = 0

            # 実際の建設フェーズの処理
            if pending:
                cmd = pending[0]
                parts = cmd.split()
                cmd_type = parts[0]
                r = int(parts[1])
                c = int(parts[2])
                cost = 0
                if cmd_type == "0":
                    cost = COST_STATION
                elif built[r][c] == 0:
                    cost = COST_RAIL
                elif built[r][c] == 2:
                    cmd = f"0 {r} {c}"
                    cmd_type = "0"
                    cost = COST_STATION
                
                if built[r][c] != 1 and funds >= cost:
                    if cmd_type == "0":
                        stations.add((r, c))
                        built[r][c] = 1
                    else:
                        built[r][c] = 2
                    funds -= cost
                    pending.pop(0)
                    output_commands.append(cmd)
                else:
                    output_commands.append("-1")
            else:
                output_commands.append("-1")
            
            # 更新処理：人物の処理が完了した場合
            if not pending and current_person_income is not None:
                connected_incomes += current_person_income
                dsu.merge(index(home, N), index(work, N))
                for group in dsu.groups():
                    connections[dsu.leader(group[0])] = group
                current_person_income = None
            funds += connected_incomes
        
        if funds > best_score:
            best_score = funds
            best_commands = output_commands
        if timeout:
            break
    
    sys.stdout.write("\n".join(best_commands[:T]))


if __name__ == "__main__":
    main()
