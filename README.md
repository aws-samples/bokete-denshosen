# ボケて電笑戦 (bokete DENSHOSEN) Workshop
This repository contains sample notebooks to train and deploy boke AI (funny AI for image captioning) model using data provided by [bokete](https://bokete.jp/). 

このリポジトリでは、ボケて電笑戦で利用されたボケ缶データセットの解説と、サンプルモデルを公開しています。
お持ちの AWS アカウント上で、Amazon SageMaker を使ったデータの前処理と、ボケ AI モデルのトレーニング・デプロイをお試しいただけます。
電笑戦に興味のあるビルダー (エンジニア、リサーチャー、データサイエンティスト) の方は、是非オリジナルなボケ AI 作りに挑戦してみて下さい！

## ボケて電笑戦について

AI は人を笑わせられるのか？「ボケて電笑戦」はその疑問に挑戦する AI 大喜利対決です。
国内最大級のお笑いメディア「[ボケて](https://bokete.jp/)」に蓄積された約 100 万を超えるボケデータ (お題画像・ボケテキストのペア) を利用して、人間には思いもよらない新たな笑いを AI が作り出せるのか競い合うのが「ボケて電笑戦」です。
新時代の笑いをテクノロジーで切り開くという壮大なチャレンジを、皆さんも楽しみませんか。

ボケて電笑戦の概要については、こちらの [紹介動画 (約1分半)](https://www.youtube.com/watch?v=u9Yt6j1tq4s) もご覧ください。詳細は、ブログ連載「電笑戦 ~ AI は人を笑わせられるのか 
[1. 挑戦を支える技術と AWS](https://aws.amazon.com/jp/builders-flash/202006/bokete/), 
[2. 電笑戦の背景と挑戦者](https://aws.amazon.com/jp/builders-flash/202007/bokete-2/), 
[3. 新たな挑戦者](https://aws.amazon.com/jp/builders-flash/202008/bokete-3/)」や、AWS Dev Day Online Japan 2021 の動画「[ボケて電笑戦技術解説 ～AIは人を笑わせられるのか？ 挑戦を支える技術とAWS～](https://www.youtube.com/watch?v=ZD9a2m5cu8o)」でご覧いただけます。

[![AWS Dev Day ボケて電笑戦](https://img.youtube.com/vi/u9Yt6j1tq4s/0.jpg)](https://www.youtube.com/watch?v=u9Yt6j1tq4s)

## ボケ缶データセットについて
Boke data: 26 GB (8.8 GB in ZIP) including 1M+ images. 

ボケて電笑戦では、ボケ缶とよばれるデータセットが用いられました。ボケ缶は株式会社オモロキにより公開されているデータセットで、ボケてのセレクトタブ (https://bokete.jp/boke/select, https://select.bokete.jp) に掲載されているボケの一部を収録したものです。ボケ缶は全部で `Blue, Yellow, Green, Red, SP` の5種類あり、星評価の数を基準に分類されています。

| 缶の種類 | 収録ボケ数 | この缶のボケについた星評価数の範囲 |
| ---- | ----: | ---- |
| blue_000 | 98,736 | 0 |
| yellow_000 | 93,762 | 1 - 100 |
| yellow_001 | 95,546 | 1 - 100 |
| yellow_002 | 96,155 | 1 - 100 |
| yellow_003 | 96,393 | 1 - 100 |
| yellow_004 | 96,464 | 1 - 100 |
| yellow_005 | 96,602 | 1 - 100 |
| yellow_006 | 98,605 | 1 - 100 |
| yellow_007 | 93,018 | 1 - 100 |
| yellow_008 | 98,117 | 1 - 100 |
| yellow_009 | 91,239 | 1 - 100 |
| green_000 | 37,342 | 101 - 1000 |
| red_000 | 8,183 | 1001 - 10000 |
| sp_000 | 380 | 10001+ |
| Total | 1,100,542 | boke |

全ボケ缶を通してボケデータの重複はありませんが、お題画像の重複はあります。
この缶の中の画像は [Flickr](https://www.flickr.com) にて Creative Commons Attribution License 2.0 ([CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)) で掲載されているもののみが収録されています。なお、この缶にボケへのコメントは含まれていません。

### ボケ缶ディレクトリ構造
- `boke.csv`: データセット本体
- `images/*.jpg`: `boke.csv` の `odai_photo_url` に対応した JPEG 形式の画像ファイル。画像サイズは長辺が 400 pixel または 600 pixel 

### CSV ヘッダ
- `id`: ボケID。https://bokete.jp/boke/:id でサイトでの表示を見ることができます。
- `odai_id`: お題ID。https://bokete.jp/odai/:odai_id でサイトでの表示を見ることができます。
- `odai_photo_id`: お題画像 ID
- `odai_photo_url`: お題画像の相対 URL
- `odai_user_id`: お題を投稿したユーザー ID
- `odai_photo_by`: お題画像のオリジナル作者
- `boke_user_id`: ボケを投稿したユーザー ID
- `text` ボケ
- `category`: `バカ・シュール・お下劣・ブラック・身内・例え・その他` からボケを投稿したユーザが選択したもの。
- `posted_at`: ボケが投稿された日時
- `rate_sum`: ボケてユーザからの星評価の合計数。ユーザは一人あたり一つのボケに一度、星1-3をつけることができる。
- `rate_count`: 星評価をしたボケてユーザの合計数。一部のケースにおいて、ユーザのアカウント削除や退会等で適切に減算されていないことがあり厳密ではない。
- `labels`: お題画像をラベル検出エンジンにかけて、一般的な物体・場所・活動・動物の種類・商品などを識別したもののリスト (順不同、`/` 区切り)。どんな画像かを識別するのにお役立てください。

## 電笑戦サンプルモデルについて
このリポジトリには、上記ボケ缶のデータからボケ AI を作るためのサンプルノートブックが含まれています。[Keras のサンプルノートブック](notebook/bokete_keras_on_sagemaker.ipynb) では、株式会社電通デジタル AIエンジニア 石川隆一氏により作成された Keras のモデルをベースに、Amazon SageMaker でのトレーニングとデプロイを試すことができます。このモデルの解説はブログ「[電笑戦 ~ AI は人を笑わせられるのか 1. 挑戦を支える技術と AWS](https://aws.amazon.com/jp/builders-flash/202006/bokete/)」をご覧ください。

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE)
file.

