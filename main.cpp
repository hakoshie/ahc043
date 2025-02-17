#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>
#include <map>
#include <random>
#include <chrono>
#include <sstream> // Include sstream for stringstream
#include <climits>  
using namespace std;

int main() {
    auto start_time = chrono::high_resolution_clock::now();

    int N, M, K, T;
    cin >> N >> M >> K >> T;

    vector<tuple<pair<int, int>, pair<int, int>>> people(M);
    for (int i = 0; i < M; ++i) {
        int r0, c0, r1, c1;
        cin >> r0 >> c0 >> r1 >> c1;
        people[i] = make_tuple(make_pair(r0, c0), make_pair(r1, c1));
    }

    sort(people.begin(), people.end(), [](const auto& a, const auto& b) {
        auto [home_a, work_a] = a;
        auto [home_b, work_b] = b;
        int dist_a = abs(home_a.first - work_a.first) + abs(home_a.second - work_a.second);
        int dist_b = abs(home_b.first - work_b.first) + abs(home_b.second - work_b.second);
        return dist_a > dist_b;
    });

    const int COST_STATION = 5000;
    const int COST_RAIL = 100;

    auto manhattan = [](pair<int, int> home, pair<int, int> work) {
        return abs(home.first - work.first) + abs(home.second - work.second);
    };

    auto find_path = [&](pair<int, int> start, pair<int, int> goal, const map<pair<int, int>, vector<pair<int, int>>>& connections, const vector<vector<int>>& built) {
        if (start == goal) {
            return vector<pair<int, int>>{start};
        }

        vector<vector<int>> dist(N, vector<int>(N, INT_MAX));
        dist[start.first][start.second] = 0;
        vector<vector<pair<int, int>>> prev(N, vector<pair<int, int>>(N, {-1, -1})); // Use {-1,-1} as sentinel value

        priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<tuple<int, int, int>>> pq;
        pq.push({0, start.first, start.second});

        vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        while (!pq.empty()) {
            auto [cost, r, c] = pq.top();
            pq.pop();

            if (make_pair(r, c) == goal) {
                vector<pair<int, int>> path;
                pair<int, int> current = {r, c};
                while (current != start) {
                    path.push_back(current);
                    current = prev[current.first][current.second];
                }
                path.push_back(start);
                reverse(path.begin(), path.end());
                return path;
            }

            vector<pair<int, int>> directions_t = directions;
            if (connections.count({r, c})) {
                for (auto [ri, ci] : connections.at({r, c})) {
                    directions_t.push_back({ri - r, ci - c});
                }
            }

            for (auto [dr, dc] : directions_t) {
                int nr = r + dr;
                int nc = c + dc;

                if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
                    int next_cost = cost;
                    if (built[nr][nc] == 1) {
                        //No cost
                    } else if (built[nr][nc] == 2) {
                        next_cost += COST_STATION;
                    } else {
                        next_cost += COST_RAIL;
                    }

                    if (next_cost < dist[nr][nc]) {
                        dist[nr][nc] = next_cost;
                        prev[nr][nc] = {r, c};
                        pq.push({next_cost, nr, nc});
                    }
                }
            }
        }

        return vector<pair<int, int>>(); // Return an empty path if no path is found
    };

    auto get_direction = [](pair<int, int> p1, pair<int, int> p2) {
        int dr = p2.first - p1.first;
        int dc = p2.second - p1.second;

        if (dr == 1) return "down";
        if (dr == -1) return "up";
        if (dc == 1) return "right";
        if (dc == -1) return "left";
        return "";
    };

    auto generate_path_commands = [&](const vector<pair<int, int>>& path, vector<vector<int>>& built) {
        vector<string> cmds;
        int L = path.size();
        int r = path[0].first;
        int c = path[0].second;

        if (built[r][c] != 1) {
            cmds.push_back("0 " + to_string(r) + " " + to_string(c));
        }

        for (int i = 1; i < L - 1; ++i) {
            pair<int, int> prev = path[i - 1];
            pair<int, int> cur = path[i];
            pair<int, int> nxt = path[i + 1];

            string d1 = get_direction(prev, cur);
            string d2 = get_direction(cur, nxt);
            if (built[cur.first][cur.second] == 1)
                continue;
            if (d1 == d2) {
                if (d1 == "left" || d1 == "right") {
                    cmds.push_back("1 " + to_string(cur.first) + " " + to_string(cur.second));
                } else {
                    cmds.push_back("2 " + to_string(cur.first) + " " + to_string(cur.second));
                }
            } else {
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
                cmds.push_back(to_string(turning_map[{d1, d2}]) + " " + to_string(cur.first) + " " + to_string(cur.second));
            }
        }

        r = path.back().first;
        c = path.back().second;
        if (built[r][c] != 1) {
            cmds.push_back("0 " + to_string(r) + " " + to_string(c));
        }
        return cmds;
    };

    auto get_detour_commands = [&](pair<int, int> home, pair<int, int> work, map<pair<int, int>, vector<pair<int, int>>>& connections,vector<vector<int>>& built) {
        vector<pair<int, int>> path = find_path(home, work, connections,built);
        if (path.empty()) return vector<string>();
        return generate_path_commands(path,built);
    };

    int best_score = INT_MIN;
    vector<string> best_commands;
    bool timeout = false;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);


    for (int trial = 0; trial < 2000; ++trial) {
        int funds = K;
        vector<vector<int>> built(N, vector<int>(N, 0));
        vector<string> output_commands;
        map<pair<int, int>, vector<pair<int, int>>> connections;
        vector<tuple<pair<int, int>, pair<int, int>>> pep_t = people;

        auto sequence = [&](int n){
            return 1;
        };

        auto sort_key = [&](const auto& x){
            auto [home,work] = x;
            int dist = manhattan(home, work)-1;
            int remaining_turns = T-output_commands.size()-1;
            int add_cost = sequence(M-pep_t.size());
            int value = dist*add_cost * 100 + 10000 - funds;
            int rest_turns = 0;
            if(value < 0){
                return (double)(dist+1) * (remaining_turns - max(dist,rest_turns));
            } else {
                return -1e9+dist;
            }
        };

        vector<string> pending_commands;
        int current_person_income = 0;
        int connected_incomes = 0;
        pair<int, int> home, work;

        for (int turn = 0; turn < T; ++turn) {
            auto current_time = chrono::high_resolution_clock::now();
            auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(current_time - start_time).count();
            if (elapsed_time > 2700) {
                timeout = true;
                break;
            }

            if(funds+connected_incomes*(T-turn-1)>best_score){
                best_score = funds + connected_incomes * (T - turn - 1);
                best_commands = output_commands;
                while (best_commands.size() < T) {
                    best_commands.push_back("-1");
                }
            }

            if (pending_commands.empty() && !pep_t.empty()) {
                sort(pep_t.begin(), pep_t.end(), [&](const auto& a, const auto& b) {
                    return sort_key(a)> sort_key(b);
                });
                 int idx = 0;
                    if (dis(gen) < 0.75 && pep_t.size() > 1) {
                        idx = (int)(dis(gen) * min((int)pep_t.size(), 20));
                    }


                home = get<0>(pep_t[idx]);
                work = get<1>(pep_t[idx]);
                pep_t.erase(pep_t.begin() + idx);
                vector<string> cmds = get_detour_commands(home, work, connections, built);

                if (!cmds.empty()) {
                    pending_commands = cmds;
                    current_person_income = manhattan(home, work);
                } else {
                    current_person_income = 0;
                }

            }

            if (!pending_commands.empty()) {
                string cmd = pending_commands[0];
                stringstream ss(cmd);
                int cmd_type, r, c;
                ss >> cmd_type >> r >> c;

                int cost = 0;
                if (cmd_type == 0) {
                    cost = COST_STATION;
                } else if (built[r][c] == 0) {
                    cost = COST_RAIL;
                } else if (built[r][c] == 2) {
                    cmd = "0 " + to_string(r) + " " + to_string(c);
                    cmd_type = 0;
                    cost = COST_STATION;
                }

                if (built[r][c]!=1 && funds >= cost) {

                    if (cmd_type == 0) {
                        built[r][c] = 1;
                    } else {
                        built[r][c] = 2;
                    }
                    funds -= cost;
                    pending_commands.erase(pending_commands.begin());
                    output_commands.push_back(cmd);
                } else {
                    output_commands.push_back("-1");
                }
            } else {
                output_commands.push_back("-1");
            }

             if (pending_commands.empty() && current_person_income > 0) {
                connected_incomes += current_person_income;
                 connections[home].push_back(work);
                 connections[work].push_back(home);
                current_person_income = 0;
            }
            funds += connected_incomes;
        }

          if (funds > best_score) {
              best_score = funds;
              best_commands = output_commands;
          }

        if(timeout){
            break;
        }
    }

    for (int i = 0; i < T; ++i) {
        cout << best_commands[i] << endl;
    }

    return 0;
}