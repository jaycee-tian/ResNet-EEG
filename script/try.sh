# #!/bin/bash
pid=$$
len=$(echo -n "$pid" | wc -m)
start_index=$(($len - 2))
last_three_chars=$(echo -n "$pid" | cut -c $start_index-)


echo "字符串的后3位是：$last_three_chars"
