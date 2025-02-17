#include <bits/stdc++.h>
#include <atcoder/dsu>
using namespace std;
using ll = long long;
 
// pair<int,int> 用のハッシュ
struct pair_hash {
    size_t operator()(const pair<int,int>& p) const {
        return ((size_t)p.first * 31ull) ^ ((size_t)p.second);
    }
};
 

 
// ユーティリティ：座標ペアを1次元インデックスに変換 (グリッドサイズ N)
int index(const pair<int,int>& p, int N) {
    return p.first * N + p.second;
}
 
// マンハッタン距離
int manhattan(const pair<int,int>& a, const pair<int,int>& b) {
    return abs(a.first - b.first) + abs(a.second - b.second);
}
 
// 2点間の方向を文字列で返す
string get_direction(const pair<int,int>& p1, const pair<int,int>& p2) {
    int dr = p2.first - p1.first;
    int dc = p2.second - p1.second;
    if (dr == 1) return "down";
    if (dr == -1) return "up";
    if (dc == 1) return "right";
    if (dc == -1) return "left";
    return "";
}
 
// Dijkstra による最短経路探索（built, connections を考慮）
vector<pair<int,int>> find_path(const pair<int,int>& start, const pair<int,int>& goal,
    const unordered_map<pair<int,int>, vector<pair<int,int>>, pair_hash>& connections,
    const vector<vector<int>>& built, int N, int COST_STATION, int COST_RAIL) {
    
    if (start == goal)
        return {start};
    
    const int INF = INT_MAX;
    vector<vector<int>> dist(N, vector<int>(N, INF));
    vector<vector<pair<int,int>>> prev(N, vector<pair<int,int>>(N, {-1,-1}));
    dist[start.first][start.second] = 0;
    
    // 優先度付きキュー： (コスト, (r, c))
    using state = pair<int, pair<int,int>>;
    priority_queue< state, vector<state>, greater<state> > pq;
    pq.push({0, start});
    
    vector<pair<int,int>> directions = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    
    while (!pq.empty()) {
        auto [cost, pos] = pq.top();
        pq.pop();
        int r = pos.first, c = pos.second;
        if (make_pair(r, c) == goal) {
            // 経路復元
            vector<pair<int,int>> path;
            pair<int,int> cur = goal;
            while (cur != start) {
                path.push_back(cur);
                cur = prev[cur.first][cur.second];
            }
            path.push_back(start);
            reverse(path.begin(), path.end());
            return path;
        }
        // 4方向＋接続先（connections）
        vector<pair<int,int>> directions_t = directions;
        pair<int,int> key = {r, c};
        if (connections.find(key) != connections.end() && !connections.at(key).empty()) {
            for (auto &conn : connections.at(key)) {
                directions_t.push_back({conn.first - r, conn.second - c});
            }
        }
        for (auto &d : directions_t) {
            int nr = r + d.first, nc = c + d.second;
            if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
            int next_cost = cost;
            if (built[nr][nc] == 1) {
                // 何も加算しない
            } else if (built[nr][nc] == 2) {
                next_cost += COST_STATION;  // 線路→駅はコスト COST_STATION
            } else {
                next_cost += COST_RAIL;     // 更地は COST_RAIL
            }
            if (next_cost < dist[nr][nc]) {
                dist[nr][nc] = next_cost;
                prev[nr][nc] = {r, c};
                pq.push({next_cost, {nr, nc}});
            }
        }
    }
    return {};  // 経路が見つからなかった場合
}
 
// 経路からコマンド列を生成する
vector<string> generate_path_commands(const vector<pair<int,int>>& path, vector<vector<int>>& built,
      unordered_map<pair<int,int>, int, pair_hash>& used,
      unordered_map<pair<int,int>, vector<pair<int,int>>, pair_hash>& connections, int N) {
    
    vector<string> cmds;
    int L = path.size();
    auto start = path[0];
    int r = start.first, c = start.second;
    if (built[r][c] != 1) {
        cmds.push_back("0 " + to_string(r) + " " + to_string(c)); // 駅設置
    }
    used[start]++;
    
    // 方向転換コマンドの対応表
    map<pair<string, string>, int> turning_map = {
        {{"up", "right"}, 6},
        {{"right", "up"}, 4},
        {{"up", "left"}, 3},
        {{"left", "up"}, 5},
        {{"down", "right"}, 5},
        {{"right", "down"}, 3},
        {{"down", "left"}, 4},
        {{"left", "down"}, 6}
    };
    
    for (int i = 1; i < L - 1; i++) {
        auto prevP = path[i-1];
        auto curP = path[i];
        auto nxtP = path[i+1];
        string d1 = get_direction(prevP, curP);
        string d2 = get_direction(curP, nxtP);
        
        bool skip = false;
        if (connections.find(curP) != connections.end()) {
            for (auto &p: connections[curP]) {
                if (p == nxtP) { skip = true; break; }
            }
        }
        if (!skip && connections.find(prevP) != connections.end()) {
            for (auto &p: connections[prevP]) {
                if (p == curP) { skip = true; break; }
            }
        }
        if (skip) continue;
        if (built[curP.first][curP.second] == 1){
            used[curP]+=2;
            continue;
        }
        if (d1 == d2) {
            if (d1 == "left" || d1 == "right") {
                cmds.push_back("1 " + to_string(curP.first) + " " + to_string(curP.second));  // 線路設置
            } else {
                cmds.push_back("2 " + to_string(curP.first) + " " + to_string(curP.second));  // 縦方向の線路
            }
        } else {
            auto key = make_pair(d1, d2);
            if (turning_map.find(key) != turning_map.end()) {
                cmds.push_back(to_string(turning_map[key]) + " " + to_string(curP.first) + " " + to_string(curP.second));
            }
        }
    }
    auto last = path.back();
    if (built[last.first][last.second] != 1) {
        cmds.push_back("0 " + to_string(last.first) + " " + to_string(last.second)); // 駅設置
    }
    used[last]++;
    return cmds;
}
 
// detour 経路のコマンド群を取得する
vector<string> get_detour_commands(const pair<int,int>& home, const pair<int,int>& work,
      unordered_map<pair<int,int>, vector<pair<int,int>>, pair_hash>& connections,
      vector<vector<int>>& built, unordered_map<pair<int,int>, int, pair_hash>& used,
      int N, int COST_STATION, int COST_RAIL) {
    
    auto path = find_path(home, work, connections, built, N, COST_STATION, COST_RAIL);
    if (path.empty()) return {};
    return generate_path_commands(path, built, used, connections, N);
}
 
// メイン関数
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int N, M, K, T;
    cin >> N >> M >> K >> T;
    
    // 人物（移動対象）の情報：((r0, c0), (r1, c1))
    vector< pair< pair<int,int>, pair<int,int> > > people;
    for (int i = 0; i < M; i++){
        int r0, c0, r1, c1;
        cin >> r0 >> c0 >> r1 >> c1;
        people.push_back({{r0, c0}, {r1, c1}});
    }
    // マンハッタン距離が大きい順にソート
    sort(people.begin(), people.end(), [&](auto &a, auto &b){
        int d1 = abs(a.first.first - a.second.first) + abs(a.first.second - a.second.second);
        int d2 = abs(b.first.first - b.second.first) + abs(b.first.second - b.second.second);
        return d1 > d2;
    });
    
    const int COST_STATION = 5000;
    const int COST_RAIL = 100;
    
    // マンハッタン距離2以内の(dx, dy)リスト（駅設置場所探索用）
    vector<pair<int,int>> moves = { {0,0}, {1,0}, {-1,0}, {0,1}, {0,-1},
                                    {2,0}, {-2,0}, {0,2}, {0,-2},
                                    {1,1}, {-1,-1}, {1,-1}, {-1,1} };
    
    auto start_time = chrono::high_resolution_clock::now();
    ll best_score = -LLONG_MAX;
    vector<string> best_commands;
    bool timeout = false;
    
    mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());
 
    // trial ループ
    for (int trial = 0; trial < 2000; trial++) {
        int funds = K;
        // built: 0 = 更地, 1 = 駅, 2 = 線路
        vector<vector<int>> built(N, vector<int>(N, 0));
        unordered_map<pair<int,int>, int, pair_hash> used;
        unordered_map<pair<int,int>, vector<pair<int,int>>, pair_hash> connections;
 
        atcoder::dsu dsu(N * N);
 
        auto pep_t = people; // コピー
        vector<string> output_commands;
        vector<string> pending;
        int current_person_income = -1; // -1: None
        int connected_incomes = 0;
        pair<int,int> chosen_home = {-1, -1}, chosen_work = {-1, -1};
 
        for (int turn = 0; turn < T; turn++) {
            auto current_time = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(current_time - start_time).count();
            if (elapsed > 2.9) { timeout = true; break; }
 
            if ( (ll)funds + (ll)connected_incomes * (T - turn - 1) > best_score ) {
                vector<string> output_commands_tmp = output_commands;
                int funds_tmp = funds + connected_incomes * (T - turn - 1);
                while ((int)output_commands_tmp.size() < T)
                    output_commands_tmp.push_back("-1");
                best_score = funds_tmp;
                best_commands = output_commands_tmp;
            }
 
            if (pending.empty() && !pep_t.empty()) {
                // sort_key をラムダで定義（各候補毎に計算）
                auto sort_key = [&](const pair< pair<int,int>, pair<int,int> > &x) -> ll {
                    auto home = x.first;
                    auto work = x.second;
                    int station_exist = 0;
                    for (auto &mv: moves) {
                        int nx = home.first + mv.first;
                        int ny = home.second + mv.second;
                        if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                            if (used[{nx, ny}] < 4 && built[nx][ny] == 1) {
                                station_exist++;
                                home = {nx, ny};
                                break;
                            }
                        }
                    }
                    for (auto &mv: moves) {
                        int nx = work.first + mv.first;
                        int ny = work.second + mv.second;
                        if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                            if (used[{nx, ny}] < 4 && built[nx][ny] == 1) {
                                station_exist++;
                                work = {nx, ny};
                                break;
                            }
                        }
                    }
                    int dist = abs(home.first - work.first) + abs(home.second - work.second) - 1;
                    int remaining_turns = T - turn - 1;
                    int cost = dist * 100 + 10000 - station_exist * 5000;
                    if (dsu.same(index(home, N), index(work, N))) {
                        cost = 0;
                    }
                    int value = cost - funds - connected_incomes * remaining_turns;
                    int rest_turns = 0;
                    if (connected_incomes != 0) {
                        rest_turns = (cost - funds) / connected_incomes;
                    }
                    if (value < 0) {
                        int n = (remaining_turns - max(dist, rest_turns));
                        return (ll)(dist + 1) * n;
                    } else {
                        return -(ll)dist;
                    }
                };
 
                sort(pep_t.begin(), pep_t.end(), [&](auto &a, auto &b){
                    return sort_key(a) > sort_key(b);
                });
 
                int idx = 0;
                uniform_real_distribution<double> prob(0.0, 1.0);
                if (pep_t.size() > 1 && prob(rng) < 0.7) {
                    int upper = min((int)pep_t.size(), 10);
                    uniform_int_distribution<int> dist_idx(0, upper - 1);
                    idx = dist_idx(rng);
                }
 
                if (idx < (int)pep_t.size()) {
                    chosen_home = pep_t[idx].first;
                    chosen_work = pep_t[idx].second;
                    current_person_income = manhattan(chosen_home, chosen_work);
                    pep_t.erase(pep_t.begin() + idx);
 
                    // 近くに既に駅が built されているか探す（home）
                    for (auto &mv: moves) {
                        int nx = chosen_home.first + mv.first;
                        int ny = chosen_home.second + mv.second;
                        if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                            if (used[{nx, ny}] < 4 && built[nx][ny] == 1) {
                                chosen_home = {nx, ny};
                                break;
                            }
                        }
                    }
                    // (work)
                    for (auto &mv: moves) {
                        int nx = chosen_work.first + mv.first;
                        int ny = chosen_work.second + mv.second;
                        if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                            if (used[{nx, ny}] < 4 && built[nx][ny] == 1) {
                                chosen_work = {nx, ny};
                                break;
                            }
                        }
                    }
 
                    // 駅設置場所をランダムに決定（まだ built されていないなら）
                    while (true) {
                        if (built[chosen_home.first][chosen_home.second] == 1) break;
                        uniform_int_distribution<int> move_dist(0, moves.size() - 1);
                        auto mv = moves[move_dist(rng)];
                        int nx = chosen_home.first + mv.first;
                        int ny = chosen_home.second + mv.second;
                        if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                            chosen_home = {nx, ny};
                            break;
                        }
                    }
                    while (true) {
                        if (built[chosen_work.first][chosen_work.second] == 1) break;
                        uniform_int_distribution<int> move_dist(0, moves.size() - 1);
                        auto mv = moves[move_dist(rng)];
                        int nx = chosen_work.first + mv.first;
                        int ny = chosen_work.second + mv.second;
                        if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                            chosen_work = {nx, ny};
                            break;
                        }
                    }
                    if (dsu.same(index(chosen_home, N), index(chosen_work, N))) {
                        connected_incomes += current_person_income;
                        current_person_income = -1;
                        continue;
                    }
                    pending = get_detour_commands(chosen_home, chosen_work, connections, built, used, N, COST_STATION, COST_RAIL);
                } else {
                    pending.clear();
                    current_person_income = 0;
                }
            }
 
            if (!pending.empty()) {
                string cmd = pending.front();
                stringstream ss(cmd);
                string cmd_type;
                int r, c;
                ss >> cmd_type >> r >> c;
                int cost = 0;
                if (cmd_type == "0") {  // 駅設置
                    cost = COST_STATION;
                } else if (built[r][c] == 0) {  // 更地→線路
                    cost = COST_RAIL;
                } else if (built[r][c] == 2) {  // 線路→駅
                    cmd = "0 " + to_string(r) + " " + to_string(c);
                    cmd_type = "0";
                    cost = COST_STATION;
                }
                if (built[r][c] != 1 && funds >= cost) {
                    if (cmd_type == "0") {
                        built[r][c] = 1;
                    } else {
                        built[r][c] = 2;
                    }
                    funds -= cost;
                    pending.erase(pending.begin());
                    output_commands.push_back(cmd);
                } else {
                    output_commands.push_back("-1");
                }
            } else {
                output_commands.push_back("-1");
            }
 
            if (pending.empty() && current_person_income != -1) {
                connected_incomes += current_person_income;
                dsu.merge(index(chosen_home, N), index(chosen_work, N));
                connections[chosen_home].push_back(chosen_work);
                connections[chosen_work].push_back(chosen_home);
                current_person_income = -1;
            }
            funds += connected_incomes;
        } // turn ループ終わり
 
        if (funds > best_score) {
            best_score = funds;
            best_commands = output_commands;
        }
        if (timeout) break;
    } // trial ループ終わり
 
    // T ターン分のコマンドを出力
    for (int i = 0; i < T && i < (int)best_commands.size(); i++) {
        cout << best_commands[i] << "\n";
    }
 
    return 0;
}
