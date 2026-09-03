# Sample gallery — the temperature and top-k dials

All samples use the **same prompt** (`'Once upon a time'`), the **same model** (`checkpoints/best.pt`, iter 4750, val loss 1.88), and the **same seed** (7) — so any difference comes purely from the sampling settings. See INTERNAL_NOTES §3.11 for the theory.

> Reproduce any row with, e.g.:
> ```
> python -m src.generate --prompt "Once upon a time" --temperature 0.7 --top-k 200 --seed 7
> ```


## Temperature 0.2, top-k 200 — conservative

*Sharpened distribution: the model almost always takes its top pick. Safe and grammatical, but repetitive and a bit flat.*

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary cloud in the sky. She was scared and didn't know what to do. Lily's mommy told her that the cloud was going to be dangerous. Lily didn't listen and went to the cloud. She saw a big cloud of white clouds and a big cloud. She thought it was so pretty. Lily's mommy told her that the cloud was not safe and that it was just a cloud. Lily felt better and continued to play in the rain. She was happy that she could help the cloud and to be safe.


## Temperature 0.7, top-k 200 — balanced (recommended)

*A good middle ground for TinyStories: coherent yet with some variety and surprise.*

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big ball, but she didn't know how to play. Lily's mommy showed her how to play with the ball. Lily was so happy and played with it all day long. But then, her little brother fell and hurt his arm. Lily's mommy told her to be careful and not to break it. Lily didn't want to be playing with the ball, but her mommy said it was important to be careful and to keep it safe. As they were walking home, Lily showed her mommy the broken ball and her mommy said it was okay to take it back. Lily learned that it's important to be careful and to always be careful and not to throw things that don't belong to you.


## Temperature 1.0, top-k 200 — creative

*The model's raw distribution (still filtered to the top 200 tokens). More varied word choice and plot turns, occasionally a little less coherent.*

> Once upon a time, there was a boy named Tom. Tom loved to study. He played everywhere he went. One day, Tom met a girl named Sally.Mary loved to play with her toys. She would read her favorite book about Tom. The girl liked to look at her many things. She liked to learn from Tom. One day, Sue heard a loud clap. It was her friend, Mia. Mia asked Tom to help her learn. Tom liked Mia's idea. They both took turns to be friends. Tom did not like the loud roar. Tom and Mia became friends with them in the town. They showed each other their ideas. "You look funny," Tom said. Tom's friends were happy too. They all studied the lot of his intelligent hands. Finally, Tom learned that everyone is laugh together. They became good friends, and the best of friends.


## Temperature 1.0, NO top-k — unfiltered tail

*No top-k safety filter, so the long tail of low-probability tokens can be sampled. Notice it drifts into odd words / less coherence — this is exactly why top-k helps.*

> Once upon a time, there was a boy named Tom. Tom loved to study. He played everywhere he went. One day, Tom met a girl named Sally.Mary loved to play with her toys. She would read her favorite book about Tom. The girl liked to look at her many things. She liked to learn from Tom. One day, Sue heard a loud clap. It was her friend, Mia. Mia asked Tom to help her learn. Tom liked Mia's idea. They both took turns to be friends. Tom did not like the loud roar. Tom and Mia became friends with them in the town. They showed each other their ideas. "You look funny," Tom said. Tom's friends were happy too. They all studied the lot of his intelligent hands. Finally, Tom learned that mc consequences makes everything her felt better and happy. They all became good friends and learned that being silly is important, and it makes everyone happy.
