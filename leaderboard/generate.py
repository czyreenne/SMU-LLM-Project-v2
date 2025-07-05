import json
import shutil
from pathlib import Path

shutil.copyfile('analysis_results.json', 'leaderboard/site/analysis_results.json')

with open('analysis_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

models = sorted(
    data,
    key=lambda x: x.get('final_score', x.get('average_score', 0)),
    reverse=True
)

for idx, model in enumerate(models, 1):
    model['rank'] = idx

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SMU LLM Project Leaderboard</title>
    <style>
    body {{
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
        background: #f7f8fa;
        color: #23272f;
        margin: 0;
        padding: 0;
    }}

    .header-section {{
        max-width: 700px;
        margin: 40px auto 0 auto;
        padding: 0 16px;
        text-align: center;
    }}

    .header-section h1 {{
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #23272f;
        letter-spacing: -1px;
    }}

    .header-section p {{
        color: #6c7280;
        font-size: 1.1rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }}

    .header-section .view-json-btn {{
        background: #f3f6fa;
        border: 1px solid #dbe0e8;
        color: #374151;
        border-radius: 8px;
        font-weight: 500;
        padding: 8px 18px;
        font-size: 1rem;
        cursor: pointer;
        transition: background 0.2s, border 0.2s;
        margin-bottom: 24px;
    }}
    .header-section .view-json-btn:hover {{
        background: #e5eaf1;
        border-color: #bfc9d7;
    }}

    .leaderboard-card {{
        background: #fff;
        border-radius: 18px;
        max-width: 1500px;
        margin: 32px auto 32px auto;
        box-shadow: 0 4px 24px rgba(32, 39, 54, 0.10);
        padding: 32px 24px 32px 24px;
        overflow-x: auto;
    }}

    #leaderboard {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 1.04rem;
        background: #fff;
        table-layout: auto;
    }}

    #leaderboard thead th {{
        background: #f6f7fb;
        color: #23272f;
        font-weight: 600;
        padding: 13px 10px;
        border-bottom: 2px solid #e3e6ee;
        text-align: left;
        font-size: 1.05rem;
        min-width: 120px;
        padding: 13px 14px;
    }}

    #leaderboard tbody tr {{
        transition: background 0.13s;
    }}

    #leaderboard tbody tr:hover {{
        background: #f3f6fa;
    }}

    #leaderboard td {{
        padding: 13px 10px;
        border-bottom: 1px solid #f0f1f6;
        vertical-align: middle;
    }}

    #leaderboard td:first-child,
    #leaderboard th:first-child {{
        text-align: right;
        min-width: 60px;
        width: 70px;
        color: #c0c4cc;
        font-weight: 700;
        font-size: 1.1rem;
    }}

    .model-name {{
        font-weight: 500;
        color: #222;
    }}

    #leaderboard th.model-name, #leaderboard td.model-name {{
        min-width: 170px;
        width: 200px;
    }}

    .rank-badge {{
        display: inline-block;
        min-width: 32px;
        height: 32px;
        line-height: 32px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 1.08rem;
        text-align: center;
        background: #f3f6fa;
        color: #6c7280;
    }}
    .rank-1 {{ background: linear-gradient(135deg, #ffe082 60%, #fffde4 100%); color: #bfa700; }}
    .rank-2 {{ background: linear-gradient(135deg, #e0e0e0 60%, #f8f9fa 100%); color: #888; }}
    .rank-3 {{ background: linear-gradient(135deg, #ffb07c 60%, #fff0e4 100%); color: #b86b00; }}

    @media (max-width: 700px) {{
        .leaderboard-card {{
            padding: 12px 2px;
        }}
        #leaderboard th, #leaderboard td {{
            font-size: 0.95rem;
            padding: 8px 6px;
        }}
    }}
    </style>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
</head>
<body>
    <div class="header-section">
        <h1>SMU LLM Project Leaderboard</h1>
        <p>
            See how leading models stack up across legal reasoning, jurisdictional understanding.<br>
            This leaderboard is updated automatically from <code>analysis_results.json</code>.
        </p>
        <button class="view-json-btn" onclick="window.open('analysis_results.json', '_blank')">
            View Raw analysis_results.json
        </button>
    </div>
    <div class="leaderboard-card">
        <table id="leaderboard" class="display">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Jurisdictional Understanding</th>
                    <th>Legal Reasoning</th>
                    <th>Comparative Analysis</th>
                    <th>Practical Application</th>
                    <th>Average Score</th>
                    <th>Overall Assessment</th>
                    <th>Final Score</th>
                </tr>
            </thead>
            <tbody>
"""

for model in models:
    try:
        scores = model['final_synthesis']['evaluation']['scores']
        avg_score = model['final_synthesis']['evaluation']['average_score']
        assessment = model['final_synthesis']['evaluation']['overall_assessment']
        final_score = model.get('final_score', '')
        rank_class = f"rank-badge rank-{model['rank']}" if model['rank'] <= 3 else "rank-badge"
        html_template += f"""
            <tr>
                <td><span class="{rank_class}">{model['rank']}</span></td>
                <td class="model-name">{model['model']}</td>
                <td>{scores.get('jurisdictional_understanding**', '')}</td>
                <td>{scores.get('legal_reasoning**', '')}</td>
                <td>{scores.get('comparative_analysis**', '')}</td>
                <td>{scores.get('practical_application**', '')}</td>
                <td>{avg_score}</td>
                <td class="overall-assessment">{assessment}</td>
                <td>{final_score}</td>
            </tr>
        """
    except Exception as e:
        print(f"Skipping model {model.get('model', '')} due to error: {e}")

html_template += """
            </tbody>
        </table>
    </div>
    <script>
        $(document).ready(function() {
            $('#leaderboard').DataTable({
                "order": [[0, "asc"]],
                "paging": false,
                "info": false,
                "searching": true
            });
        });
    </script>
</body>
</html>
"""

Path('leaderboard/site').mkdir(parents=True, exist_ok=True)
with open('leaderboard/site/index.html', 'w', encoding='utf-8') as f:
    f.write(html_template)
