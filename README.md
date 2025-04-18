<html lang="ja">
    <head>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1><center>VGGT をMulti View Stereoとして使う</center></h1>
        <h2>なにものか？</h2>
        <p>
            VGGTに複数枚の画像を入力して3Dモデルを作るプログラムです。<br>
            <br>
           ・(project) <a href="https://vgg-t.github.io/">https://vgg-t.github.io/</a><br>
           ・(paper)   <a href="https://arxiv.org/abs/2503.11651">https://arxiv.org/abs/2503.11651</a><br>
           ・(code)     <a href="https://github.com/facebookresearch/vggt">https://github.com/facebookresearch/vggt</a><br>
           ・(demo)    <a href="https://huggingface.co/spaces/facebook/vggt">https://huggingface.co/spaces/facebook/vggt</a><br>
            <br>
            (入力画像群)<br>
            <img src="images/input.png"><br>
            (得られた点群を使って3D表示)<br>
            ・何故か背景が飛び出してオブジェクトを覆ってしまう<br>
            　<img src="images/result0.gif"><br>
            ・拡大すると中に3Dもモデルはできている<br>
            　<img src="images/result1.gif"><br>
            ・後処理でカラーキーと同じ色の点群を削除してみる<br>
            　入力画像群を雑にセグメンテーションしたので輪郭にゴミが残っている...<br>
            　<img src="images/result2.gif"><br>
            ・フィルタを掛けた点群からメッシュを作成する<br>
            　<img src="images/result3.gif"><br>
            自然画ならうまくいくかと思ったが, ところどころ飛び出してくる...<br>
            <img src="images/park.gif">
        </p>
        <h2>環境構築方法</h2>
        <p>
            ※ 『VGGTを単一画像深度推定器として使う』と同じです。<br>
　　　　<br>
            ● githubからVGGTのコードをダウンロードする<br>
            　<a href="https://github.com/facebookresearch/vggt"?>https://github.com/facebookresearch/vggt</a><br>
            　Code → Download ZIP<br>
            <br>
            ● vggt-main.zip を解凍する<br>
            <br>
            ● 学習済モデルパラメータをダウンロードし vggt-main フォルダに配置する <br>
            　<a href="https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt2">https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt (5.03GB)</a><br>
            　download をクリックする<br>
            <br>
            ● Python 動作環境を構築する<br>
            　・pip install opencv-python<br>
            　・pip install torch torchvision torchaudio <br>
            　　GPUの場合は<a href="https://pytorch.org/">https://pytorch.org/</a> に従って PyTorch 2.xをインストールする<br>
            　・pip install gradio<br>
            　・pip install trimesh<br>
            　・pip install matplotlib<br>
            　・pip install scipy<br>
            　・pip install einops<br>
            　・pip install open3d<br>
        </p>
        <h2>使い方</h2>
        <p>
            ● 画像群を入力して点群のワールド座標を取得。RGB情報と合わせてPLYファイルを作成する<br>
            　　src/vggt_multi_images.py を vggt-main フォルダにコピー<br>
            　　<br>
            　　(使い方１)<br>
            　　　python  vggt_multi_images.py (画像群へのワイルドカード(例：*.png))<br>
            　　<br>
            　　(使い方２)<br>
            　　　python  vggt_multi_images.py (画像ファイル1) (画像ファイル2) ･･･ <br>
            　　<br>
            　　出力ファイル<br>
            　　・world_points.ply：	 点群の座標と色情報。3D表示に使用する<br>
            　　<br>
            　　～以下は現状、特に使わないが念のため出力～<br>
            　　・world_points.npy：点群の座標<br>
            　　　－dtype: float64<br>
            　　　－shape: 画像枚数×AIモデル解像度(高さ)×AIモデル解像度(幅)×3(x,y,z)<br>
            　　・intrinsic.npy ：	 推定されたカメラ内部パラメータ。<br>
            　　・extrinsic.npy：	 推定されたカメラ外部パラメータ。<br>
            <br>
            　・GPUで動作させる場合(未確認)<br>
            　　 vggt_single_image.py の以下2行を変更<br>
            <br>
                　　device = "cpu"<br>
            　　　　↓<br>
                　　device = "cuda" if torch.cuda.is_available() else "cpu"<br>
            <br>
                　　dtype = torch.float32<br>
               　　　↓<br>
                　　dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16<br>
            <br>
            ●点群を表示する<br>
                　python o3d_display_ply.py world_points.ply<br>
            <br>
            ●カラーキーと同じ色の点群をフィルターする<br>
            　　python filterPLY.py (PLYファイル) <br>
        　  <br>
            ●点群からメッシュを作成する<br>
            　　python o3d_pcd_to_mesh.py (PLYファイル) <br>
        </p>
    </body>
</html>
