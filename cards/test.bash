for i in *.png ; do 
    convert "$i" "$(basename "${i/.png}")".svg
done
