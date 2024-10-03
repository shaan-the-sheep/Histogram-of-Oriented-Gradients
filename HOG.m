% Function to extract Histogram of Oriented Gradients (HOG) features
function extractedHOGFeatures = extractHOGFeatures_cw(image)
    numBins = 9;               % Number of bins in the histogram
    cellSize = 8;              % Size of each cell in pixels

    % Reads the input image and stores its pixels in the matrix 'image'
    image = imread(image);
    image = double(image) / 255;

    % Input validation
    % Checks if image has 3 channels (is RGB)
    % If yes, converts image to grayscale
    if size(image, 3) == 3
        image = rgb2gray(image);
    end

    % Ensure the image dimensions are divisible by 8
    % Hardcoded rows and cols to reduce error
    image = imresize(image, [8 * floor(size(image, 1) / 8), 8 * floor(size(image, 2) / 8)]);
    rows = 8;
    cols = 16;

    % Compute Gradient Vectors
    sobelOperatorX = [-1, 0, 1];
    sobelOperatorY = sobelOperatorX';

    % Compute the derivative in the x and y direction for every pixel
    gradientX = filter2(sobelOperatorX, double(image));
    gradientY = filter2(sobelOperatorY, double(image));

    % Convert the gradient vectors to polar coordinates (angle and magnitude).
    angles = atan2(gradientY, gradientX);
    magnitude = sqrt(gradientY.^2 + gradientX.^2);

    % Compute Cell Histograms
    histograms = zeros(cols, rows, numBins);

    % Initialize extractedHOGFeatures
    extractedHOGFeatures = [];

    % For each cell in the y-direction...
    for row = 0:(cols - 1)
        rowOffset = (row * cellSize) + 1;

        % For each cell in the x-direction...
        for col = 0:(rows - 1)
            colOffset = (col * cellSize) + 1;

            % Select the pixels for this cell.
            rowIndices = rowOffset : (rowOffset + cellSize - 1);
            colIndices = colOffset : (colOffset + cellSize - 1);

            % Select the angles and magnitudes for the pixels in this cell.
            cellAngles = angles(rowIndices, colIndices);
            cellMagnitudes = magnitude(rowIndices, colIndices);

            % Compute the histogram for this cell.
            histograms(row + 1, col + 1, :) = calculateWHistogram(cellMagnitudes(:), cellAngles(:), numBins);
        end
    end

    % Block Normalization
    for row = 1:(cols - 1)
        for col = 1:(rows - 1)
            blockHists = histograms(row : row + 1, col : col + 1, :);
            magnitude = norm(blockHists(:)) + 0.01;

            % Divide all of the histogram values by the magnitude to normalize them.
            normalized = blockHists / magnitude;

            % Append the normalized histograms to our descriptor vector.
            extractedHOGFeatures = [extractedHOGFeatures; normalized(:)];
        end
    end
end

% Function to calculate weighted histogram
function weightedHistogram = calculateWHistogram(gradients, directions, numberOfBins)
    binWidth = pi / numberOfBins;
    minimumDirection = 0;
    directions(directions < 0) = directions(directions < 0) + pi;

    leftBinIndices = round((directions - minimumDirection) / binWidth);
    rightBinIndices = leftBinIndices + 1;

    leftBinCenters = ((leftBinIndices - 0.5) * binWidth) - minimumDirection;
    rightWeights = directions - leftBinCenters;
    leftWeights = binWidth - rightWeights;
    rightWeights = rightWeights / binWidth;
    leftWeights = leftWeights / binWidth;

    leftBinIndices(leftBinIndices == 0) = numberOfBins;
    rightBinIndices(rightBinIndices == (numberOfBins + 1)) = 1;

    % Create an empty row vector for the weighted histogram
    weightedHistogram = zeros(1, numberOfBins);

    % For each bin index...
    for binIndex = 1:numberOfBins
        % Find the pixels with left bin == binIndex
        pixelsLeft = (leftBinIndices == binIndex);

        % For each of the selected pixels, add the gradient magnitude to bin 'binIndex',
        % weighted by the 'leftWeight' for that pixel.
        weightedHistogram(1, binIndex) = weightedHistogram(1, binIndex) + sum(leftWeights(pixelsLeft)' * gradients(pixelsLeft));

        % Find the pixels with right bin == binIndex
        pixelsRight = (rightBinIndices == binIndex);

        % For each of the selected pixels, add the gradient magnitude to bin 'binIndex',
        % weighted by the 'rightWeight' for that pixel.
        weightedHistogram(1, binIndex) = weightedHistogram(1, binIndex) + sum(rightWeights(pixelsRight)' * gradients(pixelsRight));
    end
end
