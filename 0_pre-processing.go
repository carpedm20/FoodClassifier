package main

import (
    "image"
    "image/color"
    "runtime"

    "github.com/disintegration/imaging"
)

func main() {
    // use all CPU cores for maximum performance
    runtime.GOMAXPROCS(runtime.NumCPU())

    // input files
    files := []string{"01.jpg", "02.jpg", "03.jpg"}

    for _, file := range files {
        dstImage = imaging.Crop(file, image.Rect(50, 50, 100, 100)) 
    }
}
