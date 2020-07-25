---
layout: default
title: "Colab 操作の基本"
---

<!--
source: https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92
-->

## 1. ローカル PC からファイルをアップロード

```python
from google.colab import files
uploaded = files.upload()
```

## 2. Google Drive からファイルを入手

```python
pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
```

## 3. URL を指定してファイルを読み込む

```python
!wget [URL]
```

## 4. 同一 Google アカウントに結び付けられている Google Drive からデータを読み込む

この場合話は簡単になります。以下の手順を見てください

1. Google Drive の中に `data` というフォルダがあらかじめ作ってあったとしましょう。
2. この `data` フォルダの中に前もってデータをアップロードしてあるとします。
1. `Colab` 上では以下のようにタイプします

```python
from google.colab import drive
drive.mount('/content/drive')
```

1. すると認証手続きを経て `/contents/drive/My Drive' から自分の Google Drive 上に保存されているファイルにアクセス可能になります。


1. `GitHub` から直接ファイルを読み込む
GitHub のデータセットを開いてクリック `view raw` をクリックしてください。そしてそのリンク先の URL をコピーペーストすればデータを読むことができます。

```python
import pandas as pd
url = 'an_URL_somewhere_on_the_GitHub' 
df1 = pd.read_csv(url)
```

1. pandas とは Python でデータファイルを活用するためのライブラリ
2. GitHub 上のデータファイルを示す URL 名
3. GitHub 上の URL を pandas のデータフレームに読み込む


python のコメントは 2 種類 
`#` と 三連引用符 と
三連引用符は [`docstrings`](https://www.python.org/dev/peps/pep-0257/) に使用されます。`docstrings` [PEP258](https://www.python.org/dev/peps/pep-0258/) 

