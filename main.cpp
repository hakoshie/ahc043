#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <functional> // Include for std::function

using namespace std;

const int COST_STATION = 5000;
const int COST_RAIL = 100;

int manhattan(tuple<int, int> home, tuple<int, int> work) {
    return abs(get<0>(home) - get<0>(work)) + abs(get<1>(home) - get<1>(work));
}

string get_direction(tuple<int, int> p1, tuple<int, int> p2) {
    int dr = get<0>(p2) - get<0>(p1);
    int dc = get<1>(p2) - get<1>(p1);
    if (dr == 1) return "down";
    if (dr == -1) return "up";
    if (dc == 1) return "right";
    if (dc == -1) return "left";
    return "";
}

vector<tuple<int, int>> find_path(int N, const vector<vector<int>>& built, tuple<int, int> start, tuple<int, int> goal) {
    int sr = get<0>(start);
    int sc = get<1>(start);
    //int gr = get<0>(goal);  // These are unused, so remove the warnings
    //int gc = get<1>(goal);

    if (start == goal) {
        return {start};
    }

    vector<vector<double>> dist(N, vector<double>(N, numeric_limits<double>::infinity()));
    dist[sr][sc] = 0;
    vector<vector<tuple<int, int>>> prev(N, vector<tuple<int, int>>(N));

    priority_queue<tuple<double, int, int>, vector<tuple<double, int, int>>, greater<tuple<double, int, int>>> pq;
    pq.push({0, sr, sc});

    vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    while (!pq.empty()) {
        auto [cost, r, c] = pq.top();
        pq.pop();

        if (make_tuple(r, c) == goal) {
            vector<tuple<int, int>> path;
            tuple<int, int> current = make_tuple(r, c);
            while (current != start) {
                path.push_back(current);
                current = prev[get<0>(current)][get<1>(current)];
            }
            path.push_back(start);
            reverse(path.begin(), path.end());
            return path;
        }

        for (auto [dr, dc] : directions) {
            int nr = r + dr;
            int nc = c + dc;
            if (0 <= nr && nr < N && 0 <= nc && nc < N) {
                double next_cost;
                if (built[nr][nc] == 1) {
                    next_cost = cost;
                } else if (built[nr][nc] == 2) {
                    next_cost = cost + COST_STATION;
                } else {
                    next_cost = cost + COST_RAIL;
                }

                if (next_cost < dist[nr][nc]) {
                    dist[nr][nc] = next_cost;
                    prev[nr][nc] = make_tuple(r, c);
                    pq.push({next_cost, nr, nc});
                }
            }
        }
    }

    return {}; // Empty path if no path found
}

vector<string> generate_path_commands(int N, const vector<vector<int>>& built, const vector<tuple<int, int>>& path) {
    vector<string> cmds;
    int L = path.size();
    if (L == 0) return cmds;

    int r = get<0>(path[0]);
    int c = get<1>(path[0]);

    if (built[r][c] != 1) {
        cmds.push_back("0 " + to_string(r) + " " + to_string(c));
    }

    for (int i = 1; i < L - 1; ++i) {
        tuple<int, int> prev = path[i - 1];
        tuple<int, int> cur = path[i];
        tuple<int, int> nxt = path[i + 1];

        string d1 = get_direction(prev, cur);
        string d2 = get_direction(cur, nxt);

        r = get<0>(cur);
        c = get<1>(cur);
        if (built[r][c] == 1) continue;

        if (d1 == d2) {
            if (d1 == "left" || d1 == "right") {
                cmds.push_back("1 " + to_string(r) + " " + to_string(c));
            } else {
                cmds.push_back("2 " + to_string(r) + " " + to_string(c));
            }
        } else {
            int cmd_code;
            if (d1 == "up" && d2 == "right") cmd_code = 6;
            else if (d1 == "right" && d2 == "up") cmd_code = 4;
            else if (d1 == "up" && d2 == "left") cmd_code = 3;
            else if (d1 == "left" && d2 == "up") cmd_code = 5;
            else if (d1 == "down" && d2 == "right") cmd_code = 5;
            else if (d1 == "right" && d2 == "down") cmd_code = 3;
            else if (d1 == "down" && d2 == "left") cmd_code = 4;
            else if (d1 == "left" && d2 == "down") cmd_code = 6;
            else cmd_code = -1;

            cmds.push_back(to_string(cmd_code) + " " + to_string(r) + " " + to_string(c));
        }
    }

    r = get<0>(path[L - 1]);
    c = get<1>(path[L - 1]);
    if (built[r][c] != 1) {
        cmds.push_back("0 " + to_string(r) + " " + to_string(c));
    }

    return cmds;
}

vector<string> get_detour_commands(int N, const vector<vector<int>>& built, tuple<int, int> home, tuple<int, int> work) {
    vector<tuple<int, int>> path = find_path(N, built, home, work);
    if (path.empty()) {
        return {};
    }
    return generate_path_commands(N, built, path);
}

double sequence(int N) {
    return 1 + ((sqrt(N) - 1) / (sqrt(2000) - 1)) / 2;
}

int main() {
    // Seed the random number generator (only do this once)
    random_device rd;  // Obtain a seed from the operating system
    mt19937 gen(rd()); // Standard Mersenne Twister engine
    uniform_real_distribution<> prob_dist(0.0, 1.0); // Distribution for probabilities

    int N, M, K, T;
    cin >> N >> M >> K >> T;

    vector<pair<tuple<int, int>, tuple<int, int>>> people(M);
    for (int i = 0; i < M; ++i) {
        int r0, c0, r1, c1;
        cin >> r0 >> c0 >> r1 >> c1;
        people[i] = {{r0, c0}, {r1, c1}};
    }

    sort(people.begin(), people.end(), [](const auto& a, const auto& b) {
        return -manhattan(a.first, a.second) < -manhattan(b.first, b.second);
    });

    auto start_time = chrono::high_resolution_clock::now();

    long long best_score = -1e18;
    vector<string> best_commands;

    for (int trial = 0; trial < 30000; ++trial) {
        long long funds = K;
        vector<vector<int>> built(N, vector<int>(N, 0)); // 0: empty, 1: station, 2: rail
        vector<string> output_commands;

        vector<pair<tuple<int, int>, tuple<int, int>>> pep_t = people;
        sort(pep_t.begin(), pep_t.end(), [](const auto& a, const auto& b) {

            return manhattan(a.first, a.second) < manhattan(b.first, b.second);
        });

        vector<string> pending;
        int current_person_income = 0;
        long long connected_incomes = 0;
        tuple<int, int> home, work;

        for (int turn = 0; turn < T; ++turn) {
            auto current_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(current_time - start_time).count();
            if (duration > 2900) {
                trial = 1e9;
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
                // Correct scope for sort_key. Declare it *before* it's used.
                function<double(const pair<tuple<int, int>, tuple<int, int>>&)> sort_key =
                    [&](const pair<tuple<int, int>, tuple<int, int>>& x) {
                    int dist = manhattan(x.first, x.second);
                    int remaining_turns = T - turn - 1;
                    double add_cost = sequence(M - pep_t.size());
                    double value = dist * add_cost * 100 + 10000 - funds - connected_incomes * remaining_turns;
                    int turn_needed = (remaining_turns == 0) ? 0 : static_cast<int>((dist * add_cost * 100 + 10000 - funds) / remaining_turns);
                    if (value < 0) {
                        if (turn_needed < dist) {
                            return (double)dist * (remaining_turns - dist * add_cost);
                        } else {
                            return (double)dist * (remaining_turns - turn_needed);
                        }
                    } else {
                        return -1e9 + dist;
                    }
                };

                sort(pep_t.begin(), pep_t.end(), [&](const auto& a, const auto& b) {
                    return sort_key(a) > sort_key(b);
                });

                int idx = 0;
                if (prob_dist(gen) < .85 && pep_t.size() > 1) {  // Use random distribution
                    idx = gen() % min(static_cast<int>(pep_t.size()), 30); // Generate random index
                }

                if (idx < pep_t.size()) {
                    home = pep_t[idx].first;
                    work = pep_t[idx].second;
                    pep_t.erase(pep_t.begin() + idx);
                    vector<string> cmds = get_detour_commands(N, built, home, work);
                    if (!cmds.empty()) {
                        pending = cmds;
                        current_person_income = manhattan(home, work);
                    } else {
                        pending.clear();
                        current_person_income = 0;
                    }
                } else {
                    pending.clear();
                    current_person_income = 0;
                }
            }

            if (!pending.empty()) {
                string cmd = pending[0];
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

                if (built[r][c] != 1 && funds >= cost) {
                    if (cmd_type == 0) {
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

            if (pending.empty() && current_person_income != 0) {
                connected_incomes += current_person_income;
                current_person_income = 0;
            }
            funds += connected_incomes;
        }

        if (funds > best_score) {
            best_score = funds;
            best_commands = output_commands;
        }
    }

    for (int i = 0; i < T; ++i) {
        cout << best_commands[i] << endl;
    }

    return 0;
}