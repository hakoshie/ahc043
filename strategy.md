- 評価関数をexpected valueにする?
- クラスタリング、 halfを保存する?
- めんどいので探索のところで接続してるか確認する
- 毎回全員分dijkstraしてコストを正確に計算する、ワーシャルフロイドのほうがいいかも
- 駅は4カ所詰まってるか保存する？
- 接続している駅同士はコストゼロで移動できるよね
- 駅の位置をばらつかせるときに、含む点の数を最大化したいかも
- startとgoalのconnectionsの中でmanhattanが最小の組み合わせを見つけるとか、というかfind_pathあたりにバグがあるかもしれない
- 総スコアが高いよりも平均が強いまたは、relativeが強いほうがスコアが高い
- distの近似値としてnearest stationとのmanhatの和にする
- stationsはvectorでよくね
- 場所の新規性も重要かもしれないね、sort keyで
- なんで実行時間が遅くなるんだろうね <- coutしてただけだったわ
- sort_keyが正の部分まででrandom.choiceするか <- あまり変わらないか悪化
- distは取り方次第では4ぐらい改善するよね、distを短縮させる方に移動したなら、pointをあげる感じにしてもいいかもね