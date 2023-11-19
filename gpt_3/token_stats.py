import pandas as pd
import jsonlines as jsl
import seaborn as sns

import matplotlib.pyplot as plt


# Adapted from:
# https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == '__main__':
    data = []
    fnames = ["generated_1684340431.jsonl", "generated_1684343690.jsonl"]
    special_commands = []
    p_tokens, p_words, ratios = [], [], []
    for fname in fnames:
        with jsl.open(fname) as reader:
            for i, obj in enumerate(reader):
                prompt = obj["params"]["prompt"]
                prompt_tokens = obj["res"]["usage"]["prompt_tokens"]
                prompt_words = len(prompt.split())
                p_tokens.append(prompt_tokens)
                p_words.append(prompt_words)
                ratios.append(prompt_tokens / prompt_words)

    sns.set_theme(style="dark", font_scale=1.5)
    g = sns.boxplot(
        ratios,
        boxprops={"facecolor": (.4, .6, .8, .5)},
        medianprops={"color": "coral"},
        width=0.5,
        whis=[0, 100]
    )
    g.grid(True, which="both", axis='y', ls="--")
    g.set_xticklabels(["Token ratio"])
    plt.savefig("token_ratio.pdf", bbox_inches='tight')

    text_data = []
    colors = []
    sns_colors = sns.color_palette()
    CMAP = {
        'narrative': sns_colors[0],
        'descriptive': sns_colors[1],
        'argumentative': sns_colors[2]
    }
    with jsl.open("edited_gpt_with_cat_exp.jsonl") as reader:
        for i, obj in enumerate(reader):
            text_data.append({
                "Text ID": i + 1,
                "Number of words": len(obj["text"].split())
            })
            colors.append(CMAP[obj['category'].strip().lower()])
    df = pd.DataFrame.from_dict(text_data)
    
    sns.set_theme(style="dark", font_scale=2)
    g = sns.catplot(
        kind='bar',
        data=df,
        x='Text ID',
        y='Number of words',
        height=5,
        aspect=4,
        palette=colors,
        legend=True
    )
    g.tick_params(axis='x', labelsize=18, labelrotation=90)
    for ax in g.axes.flatten():
        ax.grid(axis='y')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        labels = list(CMAP.keys())
        handles = [plt.Rectangle((0,0),1,1, color=CMAP[c]) for c in labels]
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("excerpts_stats.pdf", bbox_inches='tight')

