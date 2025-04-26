# 内容
[MIND](https://learn.microsoft.com/ja-jp/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets)データセットを使って[推薦システム実践入門](https://www.oreilly.co.jp//books/9784873119663/)の手法やってみようのリポジトリ

Word2VecとLDAで検索結果を比較できるアプリ作ったので動作結果はここから見れます

https://home.quark-hardcore.com/mind-recommendation/

# 立ち上げ

## DB

OpenSearchを使ってます。ホスト自体はRaspberry Piでやっているので内容はそっちに記載。sshして`./Repositories/OpenSearch_setup/README.md`
opensearch-pyクライアントはここの./notebook/opensearch_connect.pyに諸々作ってます

OpenSearch Dashboardは192.168.0.45:5601から基本見れます

## フロント

Streamlitで作ってます。`cd ./streamlit_app`して`uv run streamlit run app.py --server.port 8501`
cdしないとDataLoaderクラスのinitがdirの関係で失敗します
