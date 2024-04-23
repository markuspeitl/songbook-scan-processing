# Book Scan Processing:

I was adding chords and chord changes for songs played with guitar and singing to a songbook with hundreds of song.
This was a very time intensive process and a lot of work went into adding this information to the book and enabled me
to more easily play and teach those songs in the future.
As i was often carrying that book around with me i got worried for losing all this work and therefore decided to
scan all pages of the booke and digitize it together with my changes and corrections.

That turned out well, but the book was old, with darkened pages, folds, had ripped binding holes, bad contrast,
correction marks and other defects.
Because of this i wrote a script to clean up all the scans and make a pdf with the clean pages and try to adjust for
nonuniformity of their appearance.
That ultimately outputs multiple pdfs for different distribution purposes (low dpi for display on websites, high dpi for print, medium dpi for mobile devices)
to choose from.

## Each image is:
- Read from the input directory
- Processed
- Compressed (+ Written)

## Image processing stages

This is done mainly by using image processing in the `process_image` function:
- **adjust_contrast**  
Nonlinear contrast adjustment, makes dark pixels darker and light pixels lighter  
- **clear_margins**  
Clears the regions around the edges where the edge of each page meets the empty rest of the scan region.  
If enabled shifts every second image in x or y direction, which can be used to normalize page content position for  
different sides of a page (scanning front and back side of a page and adjusting all backsides by a factor in order to normalize position)  
- **detect_recolor_image_layers**
Detect different "layers" of the scan and adjust their color:
  - background: white paper on which is printed or written onto --> adjusted to white
  - text: printed text -> the darkest filled structure --> adjusted to black
  - annotation: selects the color of text that was added with a pen (example blue) and makes it more visible and colors it in the output (example: convert blue pen marks to red text)
- **filter_adjust_image**  
Postprocess the cleaned image with blurring methods in order to reduce sharpness around the edges that was caused by "thresholding" in the last stage
  
## Conditions and performance:

- The input images need to be sortable by name by which the script determines the order in which they are added to the output pdfs.
- To improve how fast the cleaning of each page, compression, .etc is, the script is highly multithreaded/tasked, which speeds
up the process, when having a many processor cores.
- Further there is the option to create multiple pdfs at once with different resolutions and compression options, for different 
publishing purposes.
- The thresholding also makes the images much smaller as it discards unimportant information like the texture of the paper or 
defects, improvind the efficiancy of the image compression.
