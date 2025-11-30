# MapConverter

## セットアップ方法

### 1. uvのインストール

まず、Pythonのパッケージマネージャー [uv](https://docs.astral.sh/uv/) をインストールします。

以下のコマンドを実行してください。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 依存関係のインストール

次に、プロジェクトに必要なパッケージをインストールします。

```powershell
uv sync
```

## 使い方

変換したい画像のパスを指定して実行します。`<image_path>` の部分を実際の画像ファイルのパスに置き換えてください。

```powershell
uv run main.py <image_path>
```

### 実行例

```powershell
uv run main.py C:\Users\user\Downloads\image.jpg
```

## Special Thanks

- **[MIYUKINNGU](https://github.com/MIYUKINNGU)**

