header_files1=`find include -name "*.h" | xargs -i -P20 realpath {}`
header_files2=`find src -name "*.h" | xargs -i -P20 realpath {}`

source_files1=`find include -name "*.cpp" | xargs -i -P20 realpath {}`
source_files2=`find src -name "*.cpp" | xargs -i -P20 realpath {}`

header_files=(${header_files1[@]} ${header_files2[@]})
source_files=(${source_files1[@]} ${source_files2[@]})

echo ${header_files[@]}
echo ${source_files[@]}

pushd build

function run_include_what_you_use() {
    python /usr/local/bin/iwyu_tool.py -p ./ -j 8 -v $1 \
    | python /usr/local/bin/fix_includes.py -b --nocomments
}

for filename in ${header_files[@]}
do 
    run_include_what_you_use ${filename}
done 

for filename in ${source_files[@]}
do 
    run_include_what_you_use ${filename}
done 

popd