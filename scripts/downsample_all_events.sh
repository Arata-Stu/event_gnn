#!/bin/bash

# スクリプトの引数としてDSECのルートディレクトリを受け取る
if [ -z "$1" ]; then
    echo "使用法: $0 <DSEC_ROOT>"
    exit 1
fi

DSEC_ROOT=$1

# 処理対象のシーケンスを配列に格納
sequences=()
for split in train test; do
    # ディレクトリが存在する場合のみ処理を行う
    if [ -d "$DSEC_ROOT/$split" ]; then
        # `find`を使用して1階層下にあるディレクトリのみをリストアップし、配列に追加
        while IFS= read -r -d $'\0'; do
            sequences+=("$REPLY")
        done < <(find "$DSEC_ROOT/$split" -mindepth 1 -maxdepth 1 -type d -print0)
    fi
done

# 総シーケンス数を取得して表示
total_sequences=${#sequences[@]}
echo "処理対象のシーケンス総数: $total_sequences"

# カウンターを初期化
count=0

# 各シーケンスを処理
for sequence in "${sequences[@]}"; do
    # カウンターをインクリメント
    ((count++))
    
    # 現在の進捗をログとして出力
    # `basename`でディレクトリのパスから末尾のディレクトリ名（シーケンス名）を取得
    sequence_name=$(basename "$sequence")
    echo "[$count/$total_sequences] 処理中: $sequence_name"

    # ファイルパスを定義
    infile="$sequence/events/left/events.h5"
    outfile="$sequence/events/left/events_2x_new.h5"

    # # 入力ファイルが存在するか確認
    if [ -f "$infile" ]; then
        # ダウンサンプリングを実行（--scale 2 を追加）
        python3 ./downsample_events.py --input_path "$infile" --output_path "$outfile" --scale 2
    else
        echo "  -> 警告: 入力ファイルが見つかりません: $infile"
    fi
done

echo "すべての処理が完了しました。"
