for id in "$@"; do
  f=abs_$id.html
  [ -s $f ] || curl -s -m 40 -A "Mozilla/5.0" "https://arxiv.org/abs/$id" -o $f
  t=$(grep -o 'citation_title" content="[^"]*' $f | head -1 | sed 's/.*content="//')
  a=$(grep -o 'citation_author" content="[^"]*' $f | head -3 | sed 's/.*content="//' | paste -sd';')
  d=$(grep -o 'citation_date" content="[^"]*' $f | head -1 | sed 's/.*content="//')
  echo "$id | $d | $t | $a"
done
