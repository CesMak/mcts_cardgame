for i in *.png ; do
    convert "$i" "$(basename "${i/.png}")".svg
done


# delete all files with *.png
find . -type f -iname \*.png -delete
