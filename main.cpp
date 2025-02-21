#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <tuple>
#include <algorithm>
#include <map>
#include <set>
#include <random>
#include <cmath>
#include <chrono>

#include <atcoder/dsu>

using namespace std;

using ll = long long;
using pii = pair<int, int>;
using pll = pair<ll, ll>;

// Global variables for time measurement
auto start_time = chrono::high_resolution_clock::now();
const double TIME_LIMIT = 2.9; // Time limit in seconds
int N;
int COST_STATION;
int COST_RAIL;
// Function to check if the time limit has been exceeded
bool time_exceeded() {
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    return elapsed_time.count() > TIME_LIMIT;
}


struct Person {
    pii home;
    pii work;
};

int manhattan(pii home, pii work) {
    return abs(home.first - work.first) + abs(home.second - work.second);
}

int index(pii L) {
    return L.first * N + L.second;
}

pii index_inv(int i) {
    return {i / N, i % N};
}

bool vacant_station(int r, int c, const vector<vector<int>>& built) {
    int dx[] = {0, 1, 0, -1};
    int dy[] = {1, 0, -1, 0};
    for (int i = 0; i < 4; ++i) {
        int nx = r + dx[i];
        int ny = c + dy[i];
        if (0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 0) {
            return true;
        }
    }
    return false;
}

vector<pii> find_path(pii start, pii goal, const map<int, set<int>>& connections, atcoder::dsu& dsu, const vector<vector<int>>& built) {
    if (start == goal) {
        return {start};
    }
    if (dsu.same(index(start), index(goal))) {
        return {}; // Return an empty vector for no path
    }

    vector<vector<double>> dist(N, vector<double>(N, numeric_limits<double>::infinity()));
    dist[start.first][start.second] = 0;
    vector<vector<pii>> prev(N, vector<pii>(N, {-1, -1}));

    priority_queue<tuple<double, pii>, vector<tuple<double, pii>>, greater<tuple<double, pii>>> pq;
    pq.push({0, start});

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    while (!pq.empty()) {
        double cost;
        pii curr;
        tie(cost, curr) = pq.top();
        pq.pop();

        if (curr == goal) {
            // Reconstruct the path
            vector<pii> path;
            pii r = curr;
            pii c = start;
            while (r != c) {
                path.push_back(r);
                r = prev[r.first][r.second];
                if(r == make_pair(-1, -1)){
                    return {}; // Return empty path if something is wrong with reconstruction
                }
            }
            path.push_back(start);
            reverse(path.begin(), path.end());
            return path;
        }

        vector<pii> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int leader_id = dsu.leader(index(curr));
        for (int id : connections.at(leader_id)) {
            pii neighbor = index_inv(id);
            if (neighbor != curr) {
                directions.push_back({neighbor.first - curr.first, neighbor.second - curr.second});
            }
        }

        for (pii dir : directions) {
            int nr = curr.first + dir.first;
            int nc = curr.second + dir.second;
            if (0 <= nr && nr < N && 0 <= nc && nc < N) {
                double next_cost = cost;
                if (built[nr][nc] == 7) {
                    // No additional cost for existing stations
                } else if (built[nr][nc] >=1 and built[nr][nc] <= 6) {
                    if(dir==(pii){-1,0} and (built[nr][nc]==1 or built[nr][nc]==5 or built[nr][nc]==6) and (built[curr.first][curr.second]==1 or built[curr.first][curr.second]==3 or built[curr.first][curr.second]==4 or built[curr.first][curr.second]==7 or built[curr.first][curr.second]==0)){
                        
                    }else if(dir==(pii){1,0} and (built[nr][nc]==1
                    or built[nr][nc]==3 or built[nr][nc]==4) and (built[curr.first][curr.second]==1 or built[curr.first][curr.second]==5 or built[curr.first][curr.second]==6 or built[curr.first][curr.second]==7 or built[curr.first][curr.second]==0)){
                        
                    }else if(dir==(pii){0,-1} and (built[nr][nc]==2 or built[nr][nc]==4 or built[nr][nc]==5)and (built[curr.first][curr.second]==2 or built[curr.first][curr.second]==3 or built[curr.first][curr.second]==6 or built[curr.first][curr.second]==7 or built[curr.first][curr.second]==0)){
                        
                    }else if(dir==(pii){0,1} and (built[nr][nc]==2 or built[nr][nc]==3 or built[nr][nc]==6) and (built[curr.first][curr.second]==2 or built[curr.first][curr.second]==5 or built[curr.first][curr.second]==4 or built[curr.first][curr.second]==7 or built[curr.first][curr.second]==0)){
                        
                    }else{
                        next_cost += COST_STATION;
                    }
                } else {
                    next_cost += COST_RAIL;
                }

                if (next_cost < dist[nr][nc]) {
                    dist[nr][nc] = next_cost;
                    prev[nr][nc] = curr;
                    pq.push({next_cost, {nr, nc}});
                }
            }
        }
    }

    return {}; // Return an empty vector if no path is found
}


string get_direction(pii p1, pii p2) {
    int dr = p2.first - p1.first;
    int dc = p2.second - p1.second;
    if (dr == 1) return "down";
    if (dr == -1) return "up";
    if (dc == 1) return "right";
    if (dc == -1) return "left";
    return "";
}

vector<string> generate_path_commands(const vector<pii>& path, map<int, set<int>>& connections, atcoder::dsu& dsu, vector<vector<int>>& built) {
    vector<string> cmds;
    int L = path.size();
    if (L == 0) return cmds;
    pii start = path[0];
    pii goal = path.back();
    pii r = path[0];

    if (built[r.first][r.second] != 7) {
        cmds.push_back("0 " + to_string(r.first) + " " + to_string(r.second));
    }

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

    for (int i = 1; i < L - 1; ++i) {
        pii prev_pt = path[i - 1];
        pii cur_pt = path[i];
        pii next_pt = path[i + 1];

        string d1 = get_direction(prev_pt, cur_pt);
        string d2 = get_direction(cur_pt, next_pt);

        if (built[cur_pt.first][cur_pt.second] == 7) {
            dsu.merge(index(start), index(cur_pt));
            continue;
        }

        if (d1 == d2) {
            if (d1 == "left" || d1 == "right") {
                cmds.push_back("1 " + to_string(cur_pt.first) + " " + to_string(cur_pt.second));
            } else {
                cmds.push_back("2 " + to_string(cur_pt.first) + " " + to_string(cur_pt.second));
            }
        } else {
            cmds.push_back(to_string(turning_map[{d1, d2}]) + " " + to_string(cur_pt.first) + " " + to_string(cur_pt.second));
        }
    }

    r = path.back();
    if (built[r.first][r.second] != 7) {
        cmds.push_back("0 " + to_string(r.first) + " " + to_string(r.second));
    }

    return cmds;
}

vector<string> get_detour_commands(pii home, pii work, map<int, set<int>>& connections, atcoder::dsu& dsu, vector<vector<int>>& built) {
    vector<pii> path = find_path(home, work, connections, dsu, built);
    if (path.empty()) {
        return {};
    }
    return generate_path_commands(path, connections, dsu, built);
}

multiset<pii> pep2points(const vector<Person>& pep) {
    multiset<pii> points;
    for (const auto& p : pep) {
        points.insert(p.home);
        points.insert(p.work);
    }
    return points;
}

vector<pii> get_group_members(atcoder::dsu& dsu, int x, const set<pii>& stations) {
    int leader_x = dsu.leader(x);
    vector<pii> result;
    for (const auto& station : stations) {
        if (dsu.leader(index(station)) == leader_x) {
            result.push_back(station);
        }
    }
    return result;
}

pii find_nearest_point(pii start, const vector<pii>& candidates) {
    int min_dist = numeric_limits<int>::max();
    pii nearest = start;
    for (const auto& p : candidates) {
        int dist = manhattan(start, p);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = p;
        }
    }
    return nearest;
}

int main() {
    int M, K, T;
    cin >> N >> M >> K >> T;

    vector<Person> people(M);
    multiset<pii> points_ini;
    for (int i = 0; i < M; ++i) {
        cin >> people[i].home.first >> people[i].home.second >> people[i].work.first >> people[i].work.second;
        points_ini.insert(people[i].home);
        points_ini.insert(people[i].work);
    }

    sort(people.begin(), people.end(), [&](const Person& a, const Person& b) {
        return -manhattan(a.home, a.work) < -manhattan(b.home, b.work);
    });

    COST_STATION = 5000;
    COST_RAIL = 100;

    vector<pii> moves = {
        {0, 0},
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {2, 0}, {-2, 0}, {0, 2}, {0, -2},
        {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
    };

    long long best_score = -numeric_limits<long long>::max();
    vector<string> best_commands;
    bool timeout = false;

    mt19937 rng(0); // Seed the random number generator

    for (int sim = 0; sim < 1000; ++sim) {
        long long funds = K;
        // 0: empty, 1-6: rail, 7: station
        vector<vector<int>> built(N, vector<int>(N, 0));

        vector<string> output_commands;
        map<int, set<int>> connections;
        set<pii> stations;
        atcoder::dsu dsu(N * N);
        for (int i = 0; i < N * N; ++i) {
            connections[i] = {i};
        }

        vector<Person> pep_t = people;
        sort(pep_t.begin(), pep_t.end(), [&](const Person& a, const Person& b) {
            return manhattan(a.home, a.work) < manhattan(b.home, b.work);
        });

        vector<string> pending;
        int current_person_income = 0;
        long long connected_incomes = 0;
        pii home, work;

        for (int turn = 0; turn < T; ++turn) {
            if (time_exceeded()) {
                timeout = true;
                break;
            }
            
            if (best_score < funds + connected_incomes * (T - turn - 1)) {
                vector<string> output_commands_tmp = output_commands;
                long long funds_tmp = funds + connected_incomes * (T - turn - 1);
                while (output_commands_tmp.size() < T) {
                    output_commands_tmp.push_back("-1");
                }
                best_score = funds_tmp;
                best_commands = output_commands_tmp;
            }

            if (pending.empty() && !pep_t.empty()) {

                auto sort_key = [&](const Person& x) {
                    int dist = manhattan(x.home, x.work) - 1;
                    int station_exist = 0;

                    vector<pii> home_stations;
                    vector<pii> work_stations;
                    for (auto [dx, dy] : moves) {
                        int nx = x.home.first + dx, ny = x.home.second + dy;
                        if (0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 7) {
                            home_stations.push_back({nx, ny});
                        }
                    }
                    for (auto [dx, dy] : moves) {
                        int nx = x.work.first + dx, ny = x.work.second + dy;
                        if (0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 7) {
                            work_stations.push_back({nx, ny});
                        }
                    }
                    station_exist += !home_stations.empty();
                    station_exist += !work_stations.empty();
                    int cost = dist * 100 + 10000 - station_exist * 5000;

                    for (auto home_station : home_stations) {
                        bool exit_flag = false;
                        for (auto work_station : work_stations) {
                            int dist_t = manhattan(home_station, work_station) - 1;
                            cost = min(cost, dist_t * 100 + 10000 - station_exist * 5000);
                            if (dsu.same(index(home_station), index(work_station))) {
                                cost = 0;
                                exit_flag = true;
                                break;
                            }
                        }
                        if (exit_flag) break;
                    }

                    int remaining_turns = T - turn - 1;
                    double value = (double)cost - funds - connected_incomes * remaining_turns;

                    int rest_turns = connected_incomes == 0 ? 0 : max((int)ceil(((double)cost - funds) / connected_incomes), 0);
                    if (value < 0) {
                        int n = remaining_turns - max(dist, rest_turns);
                        return (double)(dist + 1) * n / ((double)cost + 1);
                    } else {
                        return (double)-(dist + 1);
                    }
                };

                vector<int> delete_ids;
                for (int i = 0; i < pep_t.size(); ++i) {
                    vector<pii> home_stations;
                    vector<pii> work_stations;
                    for (auto [dx, dy] : moves) {
                        int nx = pep_t[i].home.first + dx, ny = pep_t[i].home.second + dy;
                        if (0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 7) {
                            home_stations.push_back({nx, ny});
                        }
                    }
                    for (auto [dx, dy] : moves) {
                        int nx = pep_t[i].work.first + dx, ny = pep_t[i].work.second + dy;
                        if (0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 7) {
                            work_stations.push_back({nx, ny});
                        }
                    }
                    bool exit_flag = false;
                    for (auto home_station : home_stations) {
                        for (auto work_station : work_stations) {
                            if (dsu.same(index(home_station), index(work_station))) {
                                delete_ids.push_back(i);
                                exit_flag = true;
                                break;
                            }
                        }
                        if (exit_flag) break;
                    }
                }
                for (int i : delete_ids) {
                    connected_incomes += manhattan(pep_t[i].home, pep_t[i].work);
                }
                vector<Person> new_pep_t;
                for (int i = 0; i < pep_t.size(); ++i) {
                    bool del_flag = false;
                    for (int id : delete_ids) {
                        if (i == id) {
                            del_flag = true;
                            break;
                        }
                    }
                    if (!del_flag) new_pep_t.push_back(pep_t[i]);
                }
                pep_t = new_pep_t;

                sort(pep_t.begin(), pep_t.end(), [&](const Person& a, const Person& b){
                    return sort_key(b) < sort_key(a);
                });

                int idx = 0;
                if ((double)rng() / rng.max() < 0.7 && pep_t.size() > 1) {
                    idx = (int)((double)rng() / rng.max() * min((int)pep_t.size(), 10));
                }

                if (idx < pep_t.size()) {
                    home = pep_t[idx].home;
                    work = pep_t[idx].work;
                    current_person_income = manhattan(home, work);
                    pep_t.erase(pep_t.begin() + idx);

                    vector<pii> homes;
                    vector<pii> works;
                    for (auto [dx, dy] : moves) {
                        int nx = home.first + dx, ny = home.second + dy;
                        if (0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 7 && vacant_station(nx, ny, built)) {
                            homes.push_back({nx, ny});
                        }
                    }
                    for (auto [dx, dy] : moves) {
                        int nx = work.first + dx, ny = work.second + dy;
                        if (0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 7 && vacant_station(nx, ny, built)) {
                            works.push_back({nx, ny});
                        }
                    }

                    if (!homes.empty() && !works.empty()) {
                        int min_dist = numeric_limits<int>::max();
                        for (auto home_station : homes) {
                            for (auto work_station : works) {
                                int dist = manhattan(home_station, work_station);
                                if (dist < min_dist) {
                                    min_dist = dist;
                                    home = home_station;
                                    work = work_station;
                                }
                            }
                        }
                    } else if (!homes.empty()) {
                        home = find_nearest_point(work, homes);
                    } else if (!works.empty()) {
                        work = find_nearest_point(home, works);
                    }

                    multiset<pii> points = pep2points(pep_t);
                    if (built[home.first][home.second] != 7) {
                        int max_point_cnt = 0;
                        pii max_home = {-1, -1};
                        for (auto [dx, dy] : moves) {
                            int nx = home.first + dx, ny = home.second + dy;
                            int point_cnt = 0;
                            if (0 <= nx && nx < N && 0 <= ny && ny < N) {
                                for (auto [dx2, dy2] : moves) {
                                    int nx2 = nx + dx2, ny2 = ny + dy2;
                                    if (points_ini.count({nx2, ny2})) {
                                        point_cnt+=points_ini.count({nx2, ny2});
                                        if (points.count({nx2, ny2})) point_cnt++;
                                    }
                                }
                            }
                            if (point_cnt > max_point_cnt) {
                                max_point_cnt = point_cnt;
                                max_home = {nx, ny};
                            }
                        }
                        if (max_home.first != -1) home = max_home;
                    }

                    if (built[work.first][work.second] != 7) {
                        int max_point_cnt = 0;
                        pii max_work = {-1, -1};
                        for (auto [dx, dy] : moves) {
                            int nx = work.first + dx, ny = work.second + dy;
                            int point_cnt = 0;
                            if (0 <= nx && nx < N && 0 <= ny && ny < N) {
                                for (auto [dx2, dy2] : moves) {
                                    int nx2 = nx + dx2, ny2 = ny + dy2;
                                    if (points_ini.count({nx2, ny2})) {
                                        point_cnt++;
                                        if (points.count({nx2, ny2})) point_cnt++;
                                    }
                                }
                            }
                            if (point_cnt > max_point_cnt) {
                                max_point_cnt = point_cnt;
                                max_work = {nx, ny};
                            }
                        }
                        if (max_work.first != -1) work = max_work;
                    }

                    if (built[home.first][home.second] == 7 || built[work.first][work.second] == 7) {
                        if (built[home.first][home.second] == 7 && built[work.first][work.second] == 7) {
                            vector<pii> home_candidates = get_group_members(dsu, index(home), stations);
                            vector<pii> work_candidates = get_group_members(dsu, index(work), stations);
                            tuple<int, pii, pii> best_pair = {numeric_limits<int>::max(), {-1, -1}, {-1, -1}};
                            for (auto home_station : home_candidates) {
                                for (auto work_station : work_candidates) {
                                    best_pair = min(best_pair, {manhattan(home_station, work_station), home_station, work_station});
                                }
                            }
                            tie(ignore, home, work) = best_pair;
                        } else {
                            if (built[work.first][work.second] == 7) swap(home, work);
                            home = find_nearest_point(work, get_group_members(dsu, index(home), stations));
                        }
                    }

                    pending = get_detour_commands(home, work, connections, dsu, built);
                } else {
                    pending = {};
                }
            }
        
            if (!pending.empty()) {
                string cmd = pending[0];
                vector<string> parts;
                string current_part;
                for (char c : cmd) {
                    if (c == ' ') {
                        parts.push_back(current_part);
                        current_part = "";
                    } else {
                        current_part += c;
                    }
                }
                parts.push_back(current_part);

                string cmd_type = parts[0];
                int r = stoi(parts[1]);
                int c = stoi(parts[2]);
                int cost = 0;
                bool skip=false;
                if (cmd_type == "0") {
                    cost = COST_STATION;
                } else if (built[r][c] == 0) {
                    cost = COST_RAIL;
                } else if (built[r][c] <=6) {
                    if(cmd_type[0]-'0'==built[r][c]){
                        cost = 0;
                        skip=true;
                    }else{
                        cmd = "0 " + to_string(r) + " " + to_string(c);
                        cmd_type = "0";
                        cost=COST_STATION;
                    }
                }

                if (built[r][c] != 7 && funds >= cost and !skip) {
                    if (cmd_type == "0") {
                        stations.insert({r, c});
                        built[r][c] = 7;
                    } else {
                        built[r][c] = cmd_type[0] - '0';
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
        
            if (pending.empty() && current_person_income != 0) {
                connected_incomes += current_person_income;
                dsu.merge(index(home), index(work));
                for (int i = 0; i < N * N; ++i) {
                    connections[i].clear();
                    connections[dsu.leader(i)].insert(i);
                }
                for (int i = 0; i < N * N; ++i) {
                    connections[dsu.leader(i)].insert(i);
                }
                for(auto station : stations){
                    int dx[4]={0,1,0,-1};
                    int dy[4]={1,0,-1,0};
                    for(int i=0;i<4;i++){
                        int nx = station.first + dx[i];
                        int ny = station.second + dy[i];
                        if(0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 7){
                            dsu.merge(index(station), index({nx,ny}));
                        }
                    }
                }
                current_person_income = 0;
            }
            funds += connected_incomes;
        }
        if (funds > best_score) {
            best_score = funds;
            best_commands = output_commands;
        }
        if (timeout) break;
    }

    for (int i = 0; i < T; ++i) {
        cout << best_commands[i] << endl;
    }

    return 0;
}