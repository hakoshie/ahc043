#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <tuple>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_set>
#include <random>
#include <cmath>
#include <chrono>

#include <atcoder/dsu>

using namespace std;

using ll = long long;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
#define rep(i, n) for (int i = 0; i < (int)(n); i++)
#define FOR(i, a, b) for (int i = (int)(a); i < (int)(b); i++)
#define all(x) (x).begin(), (x).end()

template <class T> bool chmax(T &a, const T &b) { if (a < b) { a = b; return true; } return false; }
template <class T> bool chmin(T &a, const T &b) { if (a > b) { a = b; return true; } return false; }
// Global variables for time measurement
auto start_time = chrono::high_resolution_clock::now();
const double TIME_LIMIT = 2.95; // Time limit in seconds
int N;

const int COST_STATION = 5000;
const int COST_RAIL = 100;

vector<pii> moves = {
    {0, 0},
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {2, 0}, {-2, 0}, {0, 2}, {0, -2},
    {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
};
// Function to check if the time limit has been exceeded
bool time_exceeded() {
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    return elapsed_time.count() > TIME_LIMIT;
}


struct Person {
    pii home;
    pii work;
    double sort_value;
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

// Custom hash function for pii
struct PairHash {
    size_t operator()(const pii& p) const {
        return (hash<int>()(p.first) << 1) ^ hash<int>()(p.second);
    }
};

vector<pii> find_path(pii start, pii goal, const vector<vector<int>>& connections, atcoder::dsu& dsu, const vector<vector<int>>& built, int turn) {
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
            // cout<<"#final cost "<<cost<<endl;
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
                    // if(turn==0)
                    // cout<<"#visited station"<<" "<<nr<<" "<<nc<<" next_cost:"<<next_cost<<" dist:"<<dist[nr][nc]<<endl;
                } else if (built[nr][nc] >=1 and built[nr][nc] <= 6) {
                    // int COST_USED_RAIL = 5000;
                    // if(dir==(pii){-1,0} and (built[nr][nc]==2 or built[nr][nc]==3 or built[nr][nc]==6) and (built[curr.first][curr.second]==2 or built[curr.first][curr.second]==4 or built[curr.first][curr.second]==5 or built[curr.first][curr.second]==7)){
                    //     if(built[curr.first][curr.second]==7 or built[nr][nc]==7){
                    //         next_cost+=COST_STATION;
                    //     }else{
                    //         next_cost+=COST_USED_RAIL;
                    //     }
                    //     // cout<<"# y up "<<curr.first<<" "<<curr.second<<" "<<nr<<" "<<nc<<" next_cost:"<<next_cost<<endl;
                    // }else if(dir==(pii){1,0} and (built[nr][nc]==2
                    // or built[nr][nc]==4 or built[nr][nc]==5) and (built[curr.first][curr.second]==2 or built[curr.first][curr.second]==3 or built[curr.first][curr.second]==6 or built[curr.first][curr.second]==7)){
                    //     if(built[curr.first][curr.second]==7 or built[nr][nc]==7){
                    //         next_cost+=COST_STATION;
                    //     }else{
                    //         next_cost+=COST_USED_RAIL;
                    //     }
                    //     // cout<<"# y down "<<curr.first<<" "<<curr.second<<" "<<nr<<" "<<nc<<" next_cost:"<<next_cost<<endl;
                    // }else if(dir==(pii){0,-1} and (built[nr][nc]==1 or built[nr][nc]==5 or built[nr][nc]==6)and (built[curr.first][curr.second]==1 or built[curr.first][curr.second]==3 or built[curr.first][curr.second]==4 or built[curr.first][curr.second]==7)){
                    //     if(built[curr.first][curr.second]==7 or built[nr][nc]==7){
                    //         next_cost+=COST_STATION;
                    //     }else{
                    //         next_cost+=COST_USED_RAIL;
                    //     }
                    // }else if(dir==(pii){0,1} and (built[nr][nc]==1 or built[nr][nc]==3 or built[nr][nc]==4) and (built[curr.first][curr.second]==1 or built[curr.first][curr.second]==5 or built[curr.first][curr.second]==6 or built[curr.first][curr.second]==7)){
                    //     if(built[curr.first][curr.second]==7 or built[nr][nc]==7){
                    //         next_cost+=COST_STATION;
                    //     }else{
                    //         next_cost+=COST_USED_RAIL;
                    //     }
                    // }else{
                        next_cost += COST_STATION;
                    // }
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

vector<string> generate_path_commands(const vector<pii>& path,vector<vector<int>> & connections, atcoder::dsu& dsu, vector<vector<int>>& built) {
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

vector<string> get_detour_commands(pii home, pii work, vector<vector<int>>& connections, atcoder::dsu& dsu, vector<vector<int>>& built,int sim) {
    vector<pii> path = find_path(home, work, connections, dsu, built,sim);
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

vector<pii> get_group_members(atcoder::dsu& dsu, int x, const vector<pii>& stations) {
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
// べき乗重み (p=0.5: 平方根)
double weight_power(int i, double p) {
    return 1.0 / std::pow(i, p);
}

// 対数重み (c=2)
double weight_log(int i, double c) {
    return 1.0 / std::log(i + c);
}

// 指数重み (α=0.2)
double weight_exp(int i, double alpha) {
    return std::exp(-alpha * i);
}
int weighted_random(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // 逆数の重みを計算
    std::vector<double> weights(n);
    for (int i = 0; i < n; ++i) {
        weights[i] = weight_power(i + 1, 1.2);
        // weights[i] = weight_exp(i + 1, 0.5);
        // weights[i] = weight_log(i + 1, 2);
    }

    // 重みに基づいて乱数を生成
    std::discrete_distribution<int> dist(weights.begin(), weights.end());

    return dist(gen) ; // 1-indexed にする
}

// Personオブジェクトのソート値を計算する関数
double calculate_sort_value(const Person& x, int turn, int T, long long funds, long long connected_incomes, 
    atcoder::dsu& dsu, const vector<vector<int>>& built, const vector<pii>& stations) {

    int dist_m = manhattan(x.home, x.work) - 1;
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

    int manhat1 = 1e9, manhat2 = 1e9;
    for (auto st : stations) {
        chmin(manhat1, manhattan(st, x.home));
        chmin(manhat2, manhattan(st, x.work));
    }

    int dist = dist_m;
    if (stations.size() >= 2) {
        chmin(dist, manhat1 + manhat2);
    }

    int cost = dist * 100 + 10000 - station_exist * 5000;

    for (auto home_station : home_stations) {
        for (auto work_station : work_stations) {
            int dist_t = manhattan(home_station, work_station) - 1;
            chmin(cost, dist_t * 100 + 10000 - station_exist * 5000);
            if (dsu.same(index(home_station), index(work_station))) {
                return 0;
            }
        }
    }

    int remaining_turns = T - turn - 1;
    double value = (double)cost - funds - connected_incomes * remaining_turns;
    int rest_turns = connected_incomes == 0 ? 0 : max((int)ceil(((double)cost - funds) / connected_incomes), 0);

    if (value < 0) {
        int n = remaining_turns - max(dist_m, rest_turns);
        return (double)(dist_m + 1) * n / ((double)cost + 1) *pow(manhat1+manhat2,-.5);
    } else {
        return (double)-(dist_m + 1);
    }
}

double prob(){
    return rand() / (RAND_MAX );
}

int main() {
    int M, K, T;
    cin >> N >> M >> K >> T;

    vector<Person> people(M);
    multiset<pii> points_ini;
    for (int i = 0; i < M; ++i) {
        cin >> people[i].home.first >> people[i].home.second >> people[i].work.first >> people[i].work.second;
        people[i].sort_value = 0; // 初期化
        points_ini.insert(people[i].home);
        points_ini.insert(people[i].work);
    }
    vector<ll>best_scores_t(800,-1e9);
    sort(people.begin(), people.end(), [&](const Person& a, const Person& b) {
        return -manhattan(a.home, a.work) < -manhattan(b.home, b.work);
    });


    long long best_score = -numeric_limits<long long>::max();
    vector<string> best_commands;
    int best_trial=0;
    bool timeout = false;

    mt19937 rng(0); // Seed the random number generator

    for (int sim = 0; sim < 5000; ++sim) {
        long long funds = K;
        // 0: empty, 1-6: rail, 7: station
        vector<vector<int>> built(N, vector<int>(N, 0));

        vector<string> output_commands;
        vector<vector<int>> connections(N*N);
        vector<pii> stations; // Use the custom hash
        atcoder::dsu dsu(N * N);
        for (int i = 0; i < N * N; ++i) {
            connections[i].push_back(i);
        }

        vector<Person> pep_t = people;

         // ソート値の計算と格納
        for (auto &p : pep_t) {
            p.sort_value = 0; // 初期化
        }


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
                if(funds_tmp<best_scores_t[turn]-20000){
                    turn=T;
                    break;
                }
                chmax(best_scores_t[turn],funds_tmp);
                best_score = funds_tmp;
                best_trial = sim;
                best_commands = output_commands_tmp;
            }

            if (pending.empty() && !pep_t.empty()) {
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
                // pep_tの各Personに対してソート値を計算し、格納する
                 for (auto& p : pep_t) {
                    p.sort_value = calculate_sort_value(p, turn, T, funds, connected_incomes, dsu, built, stations);
                }

                // ソートの適用部分
                sort(pep_t.begin(), pep_t.end(), [&](const Person& a, const Person& b){
                    return b.sort_value < a.sort_value;
                });


                int idx = 0;
                if ((double)rng() / rng.max() < 0.7 && pep_t.size() > 1) {
                    // idx = (int)((double)rng() / rng.max() * min((int)pep_t.size(), 30));
                    idx = weighted_random(min((int)pep_t.size(), 20));
                    if(sim-best_trial>=50){
                        idx = weighted_random(min((int)pep_t.size(), 50));
                        // idx = weighted_random(min((int)pep_t.size(), 20*(sim-best_trial)/20));
                    }
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
                        if (0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 7 ) {
                            homes.push_back({nx, ny});
                        }
                    }
                    for (auto [dx, dy] : moves) {
                        int nx = work.first + dx, ny = work.second + dy;
                        if (0 <= nx && nx < N && 0 <= ny && ny < N && built[nx][ny] == 7 ) {
                            works.push_back({nx, ny});
                        }
                    }

                    if (!homes.empty() && !works.empty()) {
                        int min_dist = numeric_limits<int>::max();
                        tuple<int, pii, pii> best_pair_tmp = {numeric_limits<int>::max(), {-1, -1}, {-1, -1}};
                        for (auto home_station : homes) {
                            for (auto work_station : works) {
                                // if(dsu.same(index(home_station), index(work_station))){
                                //     best_pair_tmp = {0, home_station, work_station};
                                //     break;
                                // }
                                best_pair_tmp = min(best_pair_tmp, {manhattan(home_station, work_station), home_station, work_station});
                            }
                        }
                        tie(ignore, home, work) = best_pair_tmp;
                    } else if (!homes.empty()) {
                        home = find_nearest_point(work, homes);
                    } else if (!works.empty()) {
                        work = find_nearest_point(home, works);
                    }

                    multiset<pii> points = pep2points(pep_t);
                    if (built[home.first][home.second] != 7) {
                        int max_point_cnt = 0;
                        vector<tuple<int,int,int>>home_cands;
                  
                        pii max_home = {-1, -1};
                
                        for (auto [dx, dy] : moves) {
                            int nx = home.first + dx, ny = home.second + dy;
                            int point_cnt = 0;
                            int dist_t=1e9;
                            // for(auto st:stations){
                            //     chmin(dist_t,manhattan(st,{nx,ny}));
                            // }
                            // // if(best_score<20000)
                            // point_cnt+=(dist_init-dist_t);
                            if (0 <= nx && nx < N && 0 <= ny && ny < N) {
                            
                                for (auto [dx2, dy2] : moves) {
                                    int nx2 = nx + dx2, ny2 = ny + dy2;
                                    if (points_ini.count({nx2, ny2})) {
                                        point_cnt+=points_ini.count({nx2, ny2});
                                        if (points.count({nx2, ny2})) point_cnt++;
                                    }
                                    
                                }
                                home_cands.push_back({nx,ny,point_cnt});
                                // if (point_cnt > max_point_cnt) {
                                //     max_point_cnt = point_cnt;
                                //     max_home = {nx, ny};
                                // }
                            }
                        }
                        if(home_cands.size()){
                            sort(home_cands.begin(),home_cands.end(),[&](const tuple<int,int,int>&a,const tuple<int,int,int>&b){
                                return get<2>(a)>get<2>(b);
                            });
                            int idx;
                            if(best_score<200000){
                            // if(sim-best_trial>10){
                                idx= weighted_random(min((int)home_cands.size(), 7));
                            }else
                            idx= weighted_random(min((int)home_cands.size(), 2));
                            home={get<0>(home_cands[idx]),get<1>(home_cands[idx])};
                        }
                        
                    }

                    if (built[work.first][work.second] != 7) {
                        int max_point_cnt = 0;
                        // int dist_init=1e9;
                        // for(auto st:stations){
                        //     chmin(dist_init,manhattan(st,home));
                        // }
                        vector<tuple<int,int,int>>work_cands;
                        pii max_work = {-1, -1};
        
                        for (auto [dx, dy] : moves) {
                            int nx = work.first + dx, ny = work.second + dy;
                            int point_cnt = 0;
                            int dist_t=1e9;
                            
                            // for(auto st:stations){
                            //     chmin(dist_t,manhattan(st,{nx,ny}));
                            // }
                            // point_cnt+=(dist_init-dist_t);
                            if (0 <= nx && nx < N && 0 <= ny && ny < N) {
                            
                                for (auto [dx2, dy2] : moves) {
                                    int nx2 = nx + dx2, ny2 = ny + dy2;
                                    if (points_ini.count({nx2, ny2})) {
                                        point_cnt++;
                                        if (points.count({nx2, ny2})) point_cnt++;
                                    }
                                }
                                
                                work_cands.push_back({nx,ny,point_cnt});
                                // if (point_cnt > max_point_cnt ) {
                                //     max_point_cnt = point_cnt;
                                //     max_work = {nx, ny};
                                // }
                            }
                        
                        }
                    
                        // if (max_work.first != -1) work = max_work;
                        if(work_cands.size()){
                            sort(work_cands.begin(),work_cands.end(),[&](const tuple<int,int,int>&a,const tuple<int,int,int>&b){
                                return get<2>(a)>get<2>(b);
                            });
                            int idx;
                            if(best_score<200000){
                            // if(sim-best_trial>10){
                                idx= weighted_random(min((int)work_cands.size(), 7));
                            }else
                            idx= weighted_random(min((int)work_cands.size(), 2));
                            work={get<0>(work_cands[idx]),get<1>(work_cands[idx])};
                        }
                    }

                    if (built[home.first][home.second] == 7 || built[work.first][work.second] == 7) {
                        if (built[home.first][home.second] == 7 && built[work.first][work.second] == 7) {
                            vector<pii> home_candidates = get_group_members(dsu, index(home), stations);
                            vector<pii> work_candidates = get_group_members(dsu, index(work), stations);
                            tuple<int, pii, pii> best_pair_tmp = {numeric_limits<int>::max(), {-1, -1}, {-1, -1}};
                            for (auto home_station : home_candidates) {
                                for (auto work_station : work_candidates) {
                                    best_pair_tmp = min(best_pair_tmp, {manhattan(home_station, work_station), home_station, work_station});
                                }
                            }
                            tie(ignore, home, work) = best_pair_tmp;
                        } else {
                            if (built[work.first][work.second] == 7) swap(home, work);
                            home = find_nearest_point(work, get_group_members(dsu, index(home), stations));
                        }
                    }

                    pending = get_detour_commands(home, work, connections, dsu, built,sim);
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
                        stations.push_back({r, c});
                        built[r][c] = 7;
                        dsu.merge(index({r, c}), index(home));
                        dsu.merge(index({r, c}), index(work));
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
                // connections=vector<vector<int>>(N*N);
                rep(i,N*N){
                    connections[i].clear();
                }
                for (int i = 0; i < N * N; ++i) {
                    connections[dsu.leader(i)].push_back(i);
                }
                current_person_income = 0;
            }
            funds += connected_incomes;
        }
        if (funds > best_score) {
            best_score = funds;
            best_commands = output_commands;
            best_trial = sim;
        }
        if (timeout) break;
    }
    
    for (int i = 0; i < T; ++i) {
        cout << best_commands[i] << endl;
    }

    return 0;
}