# web-based-system9
AIS 연동 수중 음향 시각화 웹 플랫폼

## 데이터 다운로드

**Windows**

```powershell
pip install -U huggingface_hub
hf download studentzzang/seasound-data --repo-type dataset --local-dir ./data --token <받은_토큰>
```

확인 (525개면 정상)

```powershell
(Get-ChildItem data -Recurse -Filter *.wav).Count
```

**Mac**

```bash
pip3 install -U huggingface_hub
hf download studentzzang/seasound-data --repo-type dataset --local-dir ./data --token <받은_토큰>
```

확인 (525개면 정상)

```bash
find data -name "*.wav" | wc -l
```