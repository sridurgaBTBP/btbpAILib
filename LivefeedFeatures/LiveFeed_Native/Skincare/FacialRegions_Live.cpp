#include "FacialRegions_Live.h"

std::vector<std::pair<std::string, std::vector<cv::Point>>> Skincare::FacialRegions_Live::facialZones_12Old(cv::Mat rgbImage, std::vector<cv::Point> fldPoints, std::string dir)
{
	std::vector<std::pair<std::string, std::vector<cv::Point>>> DermaRegions;
	DermaRegions.resize(12);
	try
	{
		const int foreheadLength = 15, glabellaLength = 4, noseLength = 7, underEyeLength = 8, cheekLength = 15, upperLipLength = 12, lipLength = 12, chinLength = 12;
		const int nasolabialLength = 9;
		totalRegionPoints = foreheadLength + glabellaLength + noseLength + underEyeLength + cheekLength + upperLipLength + lipLength + chinLength;

		std::vector<cv::Point> forehead(foreheadLength, cv::Point(0, 0));
		std::vector<cv::Point> glabella(glabellaLength, cv::Point(0, 0));
		std::vector<cv::Point> nose(noseLength, cv::Point(0, 0));
		std::vector<cv::Point> underEyeLeft(underEyeLength, cv::Point(0, 0));
		std::vector<cv::Point> underEyeRight(underEyeLength, cv::Point(0, 0));
		std::vector<cv::Point> cheekLeft(cheekLength, cv::Point(0, 0));
		std::vector<cv::Point> cheekRight(cheekLength, cv::Point(0, 0));
		std::vector<cv::Point> upperLip(lipLength, cv::Point(0, 0));
		std::vector<cv::Point> lip(chinLength, cv::Point(0, 0));
		std::vector<cv::Point> chin(chinLength, cv::Point(0, 0));
		std::vector<cv::Point> nasolabialLeft(nasolabialLength, cv::Point(0, 0));
		std::vector<cv::Point> nasolabialRight(nasolabialLength, cv::Point(0, 0));

		//region Forehead
		int YThresh_forehead = (int)((std::max(fldPoints[19].y, fldPoints[24].y) - fldPoints[71].y)*0.05);//Its always a +ve value
		int xThresh_glabella = (int)((fldPoints[22].x - fldPoints[21].x)*0.25);//Its always a +ve value

		forehead[0].x = fldPoints[68].x; forehead[0].y = fldPoints[68].y + YThresh_forehead;
		forehead[1].x = fldPoints[69].x; forehead[1].y = fldPoints[69].y + YThresh_forehead;
		forehead[2].x = fldPoints[70].x; forehead[2].y = fldPoints[70].y + YThresh_forehead;
		forehead[3].x = fldPoints[71].x; forehead[3].y = fldPoints[71].y + YThresh_forehead;
		forehead[4].x = fldPoints[72].x; forehead[4].y = fldPoints[72].y + YThresh_forehead;
		forehead[5].x = fldPoints[73].x; forehead[5].y = fldPoints[73].y + YThresh_forehead;
		forehead[6].x = fldPoints[74].x; forehead[6].y = fldPoints[74].y + YThresh_forehead;
		forehead[8].x = fldPoints[24].x; forehead[8].y = fldPoints[24].y - YThresh_forehead;
		forehead[7].x = (int)(0.5*(fldPoints[25].x + forehead[6].x)); forehead[7].y = (int)(0.5*(forehead[6].y + forehead[8].y));
		forehead[9].x = fldPoints[23].x; forehead[9].y = fldPoints[23].y - YThresh_forehead;
		forehead[12].x = fldPoints[20].x; forehead[12].y = fldPoints[20].y - YThresh_forehead;
		forehead[13].x = fldPoints[19].x; forehead[13].y = fldPoints[19].y - YThresh_forehead;
		forehead[14].x = (int)(0.5*(fldPoints[18].x + forehead[0].x)); forehead[14].y = (int)(0.5*(forehead[0].y + forehead[12].y));

		double ForeheadSlope = (double)(forehead[9].y - forehead[12].y) / (forehead[9].x - forehead[12].x);//no need to check denominator is zero,its always give  a positive value

		cv::Point centerpoint(0, 0);
		centerpoint.x = (forehead[9].x + forehead[12].x) / 2;
		centerpoint.y = (forehead[9].y + forehead[12].y) / 2;

		forehead[10].x = centerpoint.x + xThresh_glabella * 2;//extra curvature points to connect with glabella
		forehead[10].y = forehead[12].y + (int)(ForeheadSlope*(forehead[10].x - forehead[12].x));

		forehead[11].x = centerpoint.x - xThresh_glabella * 2;
		forehead[11].y = forehead[12].y + (int)(ForeheadSlope*(forehead[11].x - forehead[12].x));

		//endregion

		//region Glabella
		glabella[0].x = forehead[11].x; glabella[0].y = forehead[11].y;
		glabella[1].x = forehead[10].x;    glabella[1].y = forehead[10].y;

		glabella[2].x = fldPoints[27].x + xThresh_glabella;//glabella points limiting upto nose start point
		glabella[2].y = fldPoints[27].y + (int)(ForeheadSlope*(glabella[2].x - fldPoints[27].x));

		glabella[3].x = fldPoints[27].x - xThresh_glabella;
		glabella[3].y = fldPoints[27].y + (int)(ForeheadSlope*(glabella[3].x - fldPoints[27].x));

		//endregion

		//region Nose
		int XThresh_noseLeft1 = (fldPoints[28].x - fldPoints[39].x);//To avoid offset when person in rotation,used different offsets for each nose point
		int XThresh_noseRight1 = (fldPoints[42].x - fldPoints[28].x);
		int XThresh_noseLeft2 = (fldPoints[29].x - fldPoints[39].x);
		int XThresh_noseRight2 = (fldPoints[42].x - fldPoints[29].x);
		int YThresh_nose = (fldPoints[30].y - fldPoints[29].y);//To identify nose edge

		nose[0].x = glabella[3].x; nose[0].y = glabella[3].y;
		nose[6].x = glabella[2].x; nose[6].y = glabella[2].y;

		nose[1].x = fldPoints[28].x - (int)((0.6*XThresh_noseLeft1));
		nose[1].y = fldPoints[28].y + (int)(ForeheadSlope*(nose[1].x - fldPoints[28].x));

		nose[5].x = fldPoints[28].x + (int)((0.6*XThresh_noseRight1));
		nose[5].y = fldPoints[28].y + (int)(ForeheadSlope*(nose[5].x - fldPoints[28].x));

		nose[2].x = fldPoints[29].x - (int)((0.7*XThresh_noseLeft2));
		nose[2].y = fldPoints[29].y + (int)(ForeheadSlope*(nose[2].x - fldPoints[29].x));

		nose[4].x = fldPoints[29].x + (int)((0.7*XThresh_noseRight2));
		nose[4].y = fldPoints[29].y + (int)(ForeheadSlope*(nose[4].x - fldPoints[29].x));

		nose[3].x = fldPoints[29].x;
		nose[3].y = fldPoints[29].y + (int)(YThresh_nose*0.7);
		//endregion

		//region Undereyes
		int YThresh_underEye = (int)((fldPoints[8].y - fldPoints[27].y)*0.15);//Its always a +ve value
		//underEyeLeft
		int XThresh_underEyeLeft = (int)((fldPoints[39].x - fldPoints[36].x)*0.1);//Its always a +ve value

		underEyeLeft[1].x = fldPoints[40].x;
		underEyeLeft[1].y = fldPoints[40].y + (int)(0.25 * YThresh_underEye);

		underEyeLeft[2].x = fldPoints[41].x;
		underEyeLeft[2].y = fldPoints[41].y + (int)(0.25 * YThresh_underEye);

		underEyeLeft[5].x = underEyeLeft[2].x;
		underEyeLeft[5].y = underEyeLeft[2].y + (int)(0.8*YThresh_underEye);

		underEyeLeft[6].x = underEyeLeft[1].x;
		underEyeLeft[6].y = underEyeLeft[1].y + (int)(0.8*YThresh_underEye);

		underEyeLeft[0].x = fldPoints[39].x;
		underEyeLeft[0].y = fldPoints[39].y + (int)(0.15 * YThresh_underEye);

		underEyeLeft[3].x = (fldPoints[36].x + fldPoints[0].x) / 2;
		underEyeLeft[3].y = fldPoints[36].y + (int)(0.05 * YThresh_underEye);

		underEyeLeft[4].x = underEyeLeft[3].x - (2 * XThresh_underEyeLeft);
		underEyeLeft[4].y = underEyeLeft[3].y + (int)(0.6*(underEyeLeft[5].y - underEyeLeft[3].y));

		underEyeLeft[7].x = underEyeLeft[0].x + XThresh_underEyeLeft;
		underEyeLeft[7].y = underEyeLeft[0].y + (int)(0.6*(underEyeLeft[6].y - underEyeLeft[0].y));

		//underEyeRight
		int XThresh_underEyeRight = (int)((fldPoints[45].x - fldPoints[42].x)*0.1);

		underEyeRight[1].x = fldPoints[47].x;
		underEyeRight[1].y = fldPoints[47].y + (int)(0.25 * YThresh_underEye);

		underEyeRight[2].x = fldPoints[46].x;
		underEyeRight[2].y = fldPoints[46].y + (int)(0.25 * YThresh_underEye);

		underEyeRight[5].x = underEyeRight[2].x;
		underEyeRight[5].y = underEyeRight[2].y + (int)(0.8*YThresh_underEye);

		underEyeRight[6].x = underEyeRight[1].x;
		underEyeRight[6].y = underEyeRight[1].y + (int)(0.8*YThresh_underEye);

		underEyeRight[0].x = fldPoints[42].x;
		underEyeRight[0].y = fldPoints[42].y + (int)(0.15 * YThresh_underEye);

		underEyeRight[3].x = (fldPoints[45].x + fldPoints[16].x) / 2;
		underEyeRight[3].y = fldPoints[45].y + (int)(0.05 * YThresh_underEye);

		underEyeRight[4].x = underEyeRight[3].x + (2 * XThresh_underEyeRight);
		underEyeRight[4].y = underEyeRight[3].y + (int)(0.6*(underEyeRight[5].y - underEyeRight[3].y));

		underEyeRight[7].x = underEyeRight[0].x - XThresh_underEyeRight;
		underEyeRight[7].y = underEyeRight[0].y + (int)(0.6*(underEyeRight[6].y - underEyeRight[0].y));
		//endregion

		//region Upper lip
		int YThresh_upperLip = (int)((fldPoints[51].y - fldPoints[33].y)*0.1);
		upperLip[0].x = fldPoints[31].x; upperLip[0].y = fldPoints[31].y + YThresh_upperLip;
		upperLip[1].x = fldPoints[33].x; upperLip[1].y = fldPoints[33].y + YThresh_upperLip;
		upperLip[2].x = fldPoints[35].x; upperLip[2].y = fldPoints[35].y + YThresh_upperLip;
		upperLip[4].x = fldPoints[54].x; upperLip[4].y = fldPoints[54].y - YThresh_upperLip;
		upperLip[3].x = upperLip[4].x - (int)(0.3*(upperLip[4].x - upperLip[2].x)); upperLip[3].y = (upperLip[4].y + upperLip[2].y) / 2;//curvature point.
		upperLip[5].x = fldPoints[53].x; upperLip[5].y = fldPoints[53].y - YThresh_upperLip;
		upperLip[6].x = fldPoints[52].x; upperLip[6].y = fldPoints[52].y - YThresh_upperLip;
		upperLip[7].x = fldPoints[51].x; upperLip[7].y = fldPoints[51].y - YThresh_upperLip;
		upperLip[8].x = fldPoints[50].x; upperLip[8].y = fldPoints[50].y - YThresh_upperLip;
		upperLip[9].x = fldPoints[49].x; upperLip[9].y = fldPoints[49].y - YThresh_upperLip;
		upperLip[10].x = fldPoints[48].x; upperLip[10].y = fldPoints[48].y - YThresh_upperLip;
		upperLip[11].x = upperLip[10].x - (int)(0.3*(upperLip[10].x - upperLip[0].x)); upperLip[11].y = (upperLip[10].y + upperLip[0].y) / 2;//curvature point.
		//endregion

		//region Chin overlap
		int XThresh_chin = (int)((fldPoints[54].x - fldPoints[48].x)*0.1);//Its always a +ve value
		int YThresh_chin = (int)((fldPoints[8].y - fldPoints[57].y)*0.1);//Its always a +ve value
		int YThresh_chin1 = (int)((fldPoints[8].y - fldPoints[48].y)*0.1); //Its always a +ve value
		int YThresh_chin2 = (int)((fldPoints[8].y - fldPoints[54].y)*0.1);//Its always a +ve value

		chin[0].x = fldPoints[48].x - XThresh_chin; chin[0].y = fldPoints[48].y + YThresh_chin1;
		chin[1].x = fldPoints[59].x; chin[1].y = fldPoints[59].y + YThresh_chin;//these are down chin points,so used different offset for above and down points
		chin[2].x = fldPoints[58].x; chin[2].y = fldPoints[58].y + YThresh_chin;
		chin[3].x = fldPoints[57].x; chin[3].y = fldPoints[57].y + YThresh_chin;
		chin[4].x = fldPoints[56].x; chin[4].y = fldPoints[56].y + YThresh_chin;
		chin[5].x = fldPoints[55].x; chin[5].y = fldPoints[55].y + YThresh_chin;


		chin[6].x = fldPoints[54].x + XThresh_chin; chin[6].y = fldPoints[54].y + YThresh_chin2;//To avoid offset variations for straight and tilt faces,used different offsets for left & right points
		chin[7].x = fldPoints[10].x; chin[7].y = fldPoints[10].y - YThresh_chin;
		chin[8].x = fldPoints[9].x; chin[8].y = fldPoints[9].y - YThresh_chin;
		chin[9].x = fldPoints[8].x; chin[9].y = fldPoints[8].y - YThresh_chin;
		chin[10].x = fldPoints[7].x; chin[10].y = fldPoints[7].y - YThresh_chin;
		chin[11].x = fldPoints[6].x; chin[11].y = fldPoints[6].y - YThresh_chin;
		//endregion

		//endregion

		//region Cheeks
		//cheekLeft
		int XThresh_cheekLeft = (int)((fldPoints[30].x - fldPoints[2].x)*0.05);
		int XThresh_noseEdgeLeft = fldPoints[33].x - fldPoints[31].x;
		cheekLeft[0].x = underEyeLeft[7].x; cheekLeft[0].y = underEyeLeft[7].y;
		cheekLeft[1].x = underEyeLeft[6].x; cheekLeft[1].y = underEyeLeft[6].y;
		cheekLeft[2].x = underEyeLeft[5].x; cheekLeft[2].y = underEyeLeft[5].y;
		cheekLeft[3].x = underEyeLeft[4].x; cheekLeft[3].y = underEyeLeft[4].y;
		cheekLeft[4].x = fldPoints[2].x + XThresh_cheekLeft; cheekLeft[4].y = fldPoints[2].y;
		cheekLeft[5].x = fldPoints[3].x + XThresh_cheekLeft; cheekLeft[5].y = fldPoints[3].y;
		cheekLeft[6].x = fldPoints[4].x + XThresh_cheekLeft; cheekLeft[6].y = fldPoints[4].y;
		cheekLeft[7].x = fldPoints[5].x + XThresh_cheekLeft; cheekLeft[7].y = fldPoints[5].y;

		cheekLeft[8].x = chin[11].x; cheekLeft[8].y = chin[11].y;
		cheekLeft[10].x = chin[1].x; cheekLeft[10].y = chin[1].y;
		cheekLeft[9].x = cheekLeft[8].x - (int)(0.3*(cheekLeft[8].x - cheekLeft[10].x)); cheekLeft[9].y = (cheekLeft[8].y + cheekLeft[10].y) / 2;


		cheekLeft[11].x = fldPoints[48].x - XThresh_cheekLeft; cheekLeft[11].y = fldPoints[48].y;
		cheekLeft[13].x = fldPoints[31].x - (int)(0.5*XThresh_noseEdgeLeft); cheekLeft[13].y = fldPoints[31].y;
		cheekLeft[12].x = upperLip[11].x; cheekLeft[12].y = upperLip[11].y;
		cheekLeft[14].x = fldPoints[31].x - (int)(0.6*XThresh_noseEdgeLeft); cheekLeft[14].y = (int)(0.5*(cheekLeft[13].y + cheekLeft[0].y));

		//cheekRight
		int XThresh_cheekRight = (int)((fldPoints[14].x - fldPoints[30].x)*0.05);
		int XThresh_noseEdgeRight = fldPoints[35].x - fldPoints[33].x;
		cheekRight[0].x = underEyeRight[7].x; cheekRight[0].y = underEyeRight[7].y;
		cheekRight[1].x = underEyeRight[6].x; cheekRight[1].y = underEyeRight[6].y;
		cheekRight[2].x = underEyeRight[5].x; cheekRight[2].y = underEyeRight[5].y;
		cheekRight[3].x = underEyeRight[4].x; cheekRight[3].y = underEyeRight[4].y;
		cheekRight[4].x = fldPoints[14].x - XThresh_cheekRight; cheekRight[4].y = fldPoints[14].y;
		cheekRight[5].x = fldPoints[13].x - XThresh_cheekRight; cheekRight[5].y = fldPoints[13].y;
		cheekRight[6].x = fldPoints[12].x - XThresh_cheekRight; cheekRight[6].y = fldPoints[12].y;
		cheekRight[7].x = fldPoints[11].x - XThresh_cheekRight; cheekRight[7].y = fldPoints[11].y;

		cheekRight[8].x = chin[7].x; cheekRight[8].y = chin[7].y;
		cheekRight[10].x = chin[5].x; cheekRight[10].y = chin[5].y;
		cheekRight[9].x = cheekRight[8].x - (int)(0.3*(cheekRight[8].x - cheekRight[10].x)); cheekRight[9].y = (cheekRight[8].y + cheekRight[10].y) / 2;

		cheekRight[11].x = fldPoints[54].x + XThresh_cheekRight; cheekRight[11].y = fldPoints[54].y;
		cheekRight[13].x = fldPoints[35].x + (int)(0.5*XThresh_noseEdgeRight); cheekRight[13].y = fldPoints[35].y;
		cheekRight[12].x = upperLip[3].x; cheekRight[12].y = upperLip[3].y;
		cheekRight[14].x = fldPoints[35].x + (int)(0.6*XThresh_noseEdgeRight); cheekRight[14].y = (cheekRight[13].y + cheekRight[0].y) / 2;
		//endregion

		//region Lip
		lip[0].x = fldPoints[48].x; lip[0].y = fldPoints[48].y;
		lip[1].x = fldPoints[49].x; lip[1].y = fldPoints[49].y;
		lip[2].x = fldPoints[50].x; lip[2].y = fldPoints[50].y;
		lip[3].x = fldPoints[51].x; lip[3].y = fldPoints[51].y;
		lip[4].x = fldPoints[52].x; lip[4].y = fldPoints[52].y;
		lip[5].x = fldPoints[53].x; lip[5].y = fldPoints[53].y;
		lip[6].x = fldPoints[54].x; lip[6].y = fldPoints[54].y;
		lip[7].x = fldPoints[55].x; lip[7].y = fldPoints[55].y;
		lip[8].x = fldPoints[56].x; lip[8].y = fldPoints[56].y;
		lip[9].x = fldPoints[57].x; lip[9].y = fldPoints[57].y;
		lip[10].x = fldPoints[58].x; lip[10].y = fldPoints[58].y;
		lip[11].x = fldPoints[59].x; lip[11].y = fldPoints[59].y;
		//endregion

		//region Nasolabial
		//Nasolabial Left
		int XThresh_nasolabialLeft1 = (fldPoints[31].x - cheekLeft[5].x);//Its always a +ve value
		int XThresh_nasolabialLeft2 = (fldPoints[48].x - cheekLeft[6].x);//Its always a +ve value
		int XThresh_nasolabialLeft3 = (fldPoints[59].x - cheekLeft[7].x);//Its always a +ve value
		int YThresh_nasolabial = (int)((fldPoints[51].y - fldPoints[33].y)*0.3);//Its always a +ve value

		nasolabialLeft[0].x = cheekLeft[13].x; nasolabialLeft[0].y = cheekLeft[13].y;
		nasolabialLeft[1].x = cheekLeft[12].x; nasolabialLeft[1].y = cheekLeft[12].y;
		nasolabialLeft[2].x = cheekLeft[11].x; nasolabialLeft[2].y = cheekLeft[11].y;
		nasolabialLeft[3].x = cheekLeft[10].x; nasolabialLeft[3].y = cheekLeft[10].y;
		nasolabialLeft[4].x = cheekLeft[9].x; nasolabialLeft[4].y = cheekLeft[9].y;
		nasolabialLeft[5].x = nasolabialLeft[3].x - (int)((XThresh_nasolabialLeft3*0.8)); nasolabialLeft[5].y = nasolabialLeft[3].y;
		nasolabialLeft[6].x = nasolabialLeft[2].x - (int)((XThresh_nasolabialLeft2*0.5)); nasolabialLeft[6].y = nasolabialLeft[2].y;
		nasolabialLeft[7].x = nasolabialLeft[1].x - (int)((XThresh_nasolabialLeft1*0.4)); nasolabialLeft[7].y = nasolabialLeft[1].y;
		nasolabialLeft[8].x = nasolabialLeft[1].x - ((int)(XThresh_nasolabialLeft1*0.2)); nasolabialLeft[8].y = nasolabialLeft[0].y + YThresh_nasolabial;

		//Nasolabial Right
		int XThresh_nasolabialRight1 = (cheekRight[5].x - fldPoints[35].x);//Its always a +ve value
		int XThresh_nasolabialRight2 = (cheekRight[6].x - fldPoints[54].x);//Its always a +ve value
		int XThresh_nasolabialRight3 = (cheekRight[7].x - fldPoints[55].x);//Its always a +ve value

		nasolabialRight[0].x = cheekRight[13].x; nasolabialRight[0].y = cheekRight[13].y;
		nasolabialRight[1].x = cheekRight[12].x; nasolabialRight[1].y = cheekRight[12].y;
		nasolabialRight[2].x = cheekRight[11].x; nasolabialRight[2].y = cheekRight[11].y;
		nasolabialRight[3].x = cheekRight[10].x; nasolabialRight[3].y = cheekRight[10].y;

		nasolabialRight[4].x = cheekRight[9].x; nasolabialRight[4].y = cheekRight[9].y;
		nasolabialRight[5].x = nasolabialRight[3].x + (int)((XThresh_nasolabialRight3*0.8)); nasolabialRight[5].y = nasolabialRight[3].y;
		nasolabialRight[6].x = nasolabialRight[2].x + (int)((XThresh_nasolabialRight2*0.5)); nasolabialRight[6].y = nasolabialRight[2].y;
		nasolabialRight[7].x = nasolabialRight[1].x + (int)((XThresh_nasolabialRight1*0.4)); nasolabialRight[7].y = nasolabialRight[1].y;
		nasolabialRight[8].x = nasolabialRight[1].x + (int)((XThresh_nasolabialRight1*0.2)); nasolabialRight[8].y = nasolabialRight[0].y + YThresh_nasolabial;
		// endregion

		//region noseforredness
		const int noseforrednessLength = 15;
		std::vector<cv::Point> noseforredness(noseforrednessLength, cv::Point(0, 0));

		int XThresh_noseLeft3 = (fldPoints[30].x - fldPoints[39].x);//Its always a +ve value
		int XThresh_noseRight3 = (fldPoints[42].x - fldPoints[30].x);//Its always a +ve value

		int YThresh_nose1 = (int)(0.4*(fldPoints[51].y - fldPoints[33].y));

		noseforredness[0].x = nose[0].x; noseforredness[0].y = nose[0].y;
		noseforredness[1].x = nose[1].x; noseforredness[1].y = nose[1].y;
		noseforredness[2].x = nose[2].x; noseforredness[2].y = nose[2].y;

		noseforredness[12].x = nose[4].x; noseforredness[12].y = nose[4].y;
		noseforredness[13].x = nose[5].x; noseforredness[13].y = nose[5].y;
		noseforredness[14].x = nose[6].x; noseforredness[14].y = nose[6].y;

		noseforredness[3].x = fldPoints[30].x - (int)(0.8*XThresh_noseLeft3);
		noseforredness[3].y = fldPoints[30].y + (int)(ForeheadSlope*(noseforredness[3].x - fldPoints[30].x));

		noseforredness[4].x = cheekLeft[13].x; noseforredness[4].y = cheekLeft[13].y;
		noseforredness[10].x = cheekRight[13].x; noseforredness[10].y = cheekRight[13].y;

		noseforredness[11].x = fldPoints[30].x + (int)(0.8*XThresh_noseRight3);
		noseforredness[11].y = fldPoints[30].y + (int)(ForeheadSlope*(noseforredness[11].x - fldPoints[30].x));

		noseforredness[5].x = fldPoints[31].x;
		noseforredness[5].y = fldPoints[31].y + YThresh_nose1;

		noseforredness[6].x = fldPoints[32].x;
		noseforredness[6].y = fldPoints[32].y + YThresh_nose1;

		noseforredness[7].x = fldPoints[33].x;
		noseforredness[7].y = fldPoints[33].y + YThresh_nose1;

		noseforredness[8].x = fldPoints[34].x;
		noseforredness[8].y = fldPoints[34].y + YThresh_nose1;

		noseforredness[9].x = fldPoints[35].x;
		noseforredness[9].y = fldPoints[35].y + YThresh_nose1;
		//endregion

		//region cheekforredness
		const int cheekforrednessLength = 16;
		std::vector<cv::Point> cheekLeftforredness(cheekforrednessLength, cv::Point(0, 0));
		std::vector<cv::Point> cheekRightforredness(cheekforrednessLength, cv::Point(0, 0));

		//Combining cheek redness points with nose redness region end points
		cheekLeftforredness[0].x = cheekLeft[1].x; cheekLeftforredness[0].y = cheekLeft[1].y;
		cheekLeftforredness[1].x = cheekLeft[2].x; cheekLeftforredness[1].y = cheekLeft[2].y;
		cheekLeftforredness[2].x = cheekLeft[3].x; cheekLeftforredness[2].y = cheekLeft[3].y;
		cheekLeftforredness[3].x = cheekLeft[4].x; cheekLeftforredness[3].y = cheekLeft[4].y;
		cheekLeftforredness[4].x = cheekLeft[5].x; cheekLeftforredness[4].y = cheekLeft[5].y;
		cheekLeftforredness[5].x = cheekLeft[6].x; cheekLeftforredness[5].y = cheekLeft[6].y;
		cheekLeftforredness[6].x = cheekLeft[7].x; cheekLeftforredness[6].y = cheekLeft[7].y;
		cheekLeftforredness[7].x = cheekLeft[8].x; cheekLeftforredness[7].y = cheekLeft[8].y;
		cheekLeftforredness[8].x = cheekLeft[9].x; cheekLeftforredness[8].y = cheekLeft[9].y;
		cheekLeftforredness[9].x = cheekLeft[10].x; cheekLeftforredness[9].y = cheekLeft[10].y;
		cheekLeftforredness[10].x = cheekLeft[11].x; cheekLeftforredness[10].y = cheekLeft[11].y;
		cheekLeftforredness[11].x = cheekLeft[12].x; cheekLeftforredness[11].y = cheekLeft[12].y;
		cheekLeftforredness[12].x = cheekLeft[13].x; cheekLeftforredness[12].y = cheekLeft[13].y;
		cheekLeftforredness[13].x = noseforredness[3].x; cheekLeftforredness[13].y = noseforredness[3].y;
		cheekLeftforredness[14].x = noseforredness[2].x; cheekLeftforredness[14].y = noseforredness[2].y;
		cheekLeftforredness[15].x = noseforredness[1].x; cheekLeftforredness[15].y = noseforredness[1].y;

		cheekRightforredness[0].x = cheekRight[1].x; cheekRightforredness[0].y = cheekRight[1].y;
		cheekRightforredness[1].x = cheekRight[2].x; cheekRightforredness[1].y = cheekRight[2].y;
		cheekRightforredness[2].x = cheekRight[3].x; cheekRightforredness[2].y = cheekRight[3].y;
		cheekRightforredness[3].x = cheekRight[4].x; cheekRightforredness[3].y = cheekRight[4].y;
		cheekRightforredness[4].x = cheekRight[5].x; cheekRightforredness[4].y = cheekRight[5].y;
		cheekRightforredness[5].x = cheekRight[6].x; cheekRightforredness[5].y = cheekRight[6].y;
		cheekRightforredness[6].x = cheekRight[7].x; cheekRightforredness[6].y = cheekRight[7].y;
		cheekRightforredness[7].x = cheekRight[8].x; cheekRightforredness[7].y = cheekRight[8].y;
		cheekRightforredness[8].x = cheekRight[9].x; cheekRightforredness[8].y = cheekRight[9].y;
		cheekRightforredness[9].x = cheekRight[10].x; cheekRightforredness[9].y = cheekRight[10].y;
		cheekRightforredness[10].x = cheekRight[11].x; cheekRightforredness[10].y = cheekRight[11].y;
		cheekRightforredness[11].x = cheekRight[12].x; cheekRightforredness[11].y = cheekRight[12].y;
		cheekRightforredness[12].x = cheekRight[13].x; cheekRightforredness[12].y = cheekRight[13].y;
		cheekRightforredness[13].x = noseforredness[11].x; cheekRightforredness[13].y = noseforredness[11].y;
		cheekRightforredness[14].x = noseforredness[12].x; cheekRightforredness[14].y = noseforredness[12].y;
		cheekRightforredness[15].x = noseforredness[13].x; cheekRightforredness[15].y = noseforredness[13].y;
		//endregion

		// region cheekforUEC
		const int cheekforUECLength = 6;
		std::vector<cv::Point> cheekLeftforUEC(cheekforUECLength, cv::Point(0, 0));
		std::vector<cv::Point> cheekRightforUEC(cheekforUECLength, cv::Point(0, 0));

		cheekLeftforUEC[0].x = cheekLeft[0].x; cheekLeftforUEC[0].y = cheekLeft[0].y;
		cheekLeftforUEC[1].x = cheekLeft[1].x; cheekLeftforUEC[1].y = cheekLeft[1].y;
		cheekLeftforUEC[2].x = cheekLeft[2].x; cheekLeftforUEC[2].y = cheekLeft[2].y;
		cheekLeftforUEC[3].x = cheekLeft[3].x; cheekLeftforUEC[3].y = cheekLeft[3].y;
		cheekLeftforUEC[5].x = cheekLeft[13].x; cheekLeftforUEC[5].y = cheekLeft[13].y;
		cheekLeftforUEC[4].x = cheekLeftforUEC[5].x - (int)((cheekLeftforUEC[5].x - cheekLeft[5].x)*0.8);
		cheekLeftforUEC[4].y = cheekLeftforUEC[5].y + (int)(ForeheadSlope*(cheekLeftforUEC[4].x - cheekLeftforUEC[5].x));

		cheekRightforUEC[0].x = cheekRight[0].x; cheekRightforUEC[0].y = cheekRight[0].y;
		cheekRightforUEC[1].x = cheekRight[1].x; cheekRightforUEC[1].y = cheekRight[1].y;
		cheekRightforUEC[2].x = cheekRight[2].x; cheekRightforUEC[2].y = cheekRight[2].y;
		cheekRightforUEC[3].x = cheekRight[3].x; cheekRightforUEC[3].y = cheekRight[3].y;
		cheekRightforUEC[5].x = cheekRight[13].x; cheekRightforUEC[5].y = cheekRight[13].y;
		cheekRightforUEC[4].x = cheekRightforUEC[5].x + (int)((cheekRight[5].x - cheekRightforUEC[5].x)*0.8);
		cheekRightforUEC[4].y = cheekRightforUEC[5].y + (int)(ForeheadSlope*(cheekRightforUEC[4].x - cheekRightforUEC[5].x));
		// endregion


		std::vector<cv::Point> RightForeHead, LeftForeHead, MiddleForehead;
		RightForeHead.push_back(forehead[0]);
		RightForeHead.push_back(forehead[1]);
		RightForeHead.push_back(forehead[2]);
		RightForeHead.push_back(forehead[12]);
		RightForeHead.push_back(forehead[13]);
		RightForeHead.push_back(forehead[14]);

		MiddleForehead.push_back(forehead[2]);
		MiddleForehead.push_back(forehead[3]);
		MiddleForehead.push_back(forehead[4]);
		MiddleForehead.push_back(forehead[9]);
		MiddleForehead.push_back(forehead[10]);
		MiddleForehead.push_back(forehead[11]);
		MiddleForehead.push_back(forehead[12]);

		LeftForeHead.push_back(forehead[4]);
		LeftForeHead.push_back(forehead[5]);
		LeftForeHead.push_back(forehead[6]);
		LeftForeHead.push_back(forehead[7]);
		LeftForeHead.push_back(forehead[8]);
		LeftForeHead.push_back(forehead[9]);

		std::vector<cv::Point> Nose_Derma;
		Nose_Derma.push_back(noseforredness[0]);
		Nose_Derma.push_back(noseforredness[1]);
		Nose_Derma.push_back(noseforredness[2]);
		Nose_Derma.push_back(noseforredness[3]);
		Nose_Derma.push_back(cv::Point(fldPoints[30].x, fldPoints[30].y + (int)(0.5*abs(fldPoints[33].y - fldPoints[30].y))));
		Nose_Derma.push_back(noseforredness[11]);
		Nose_Derma.push_back(noseforredness[12]);
		Nose_Derma.push_back(noseforredness[13]);
		Nose_Derma.push_back(noseforredness[14]);

		std::vector<cv::Point> cheekRight_right, cheekRight_left;

		for (int k = 2; k <= 9; k++)
		{
			cheekRight_right.push_back(cheekLeft[k]);
		}
		cheekRight_right.push_back(nasolabialLeft[5]);
		cheekRight_right.push_back(nasolabialLeft[6]);
		cheekRight_right.push_back(cv::Point(nasolabialLeft[7].x + (int)(0.5*(nasolabialLeft[8].x - nasolabialLeft[7].x)), nasolabialLeft[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));

		//cheekRight_left.push_back(cheekLeft[12]);
		cheekRight_left.push_back(cheekLeft[13]);
		cheekRight_left.push_back(noseforredness[3]);
		cheekRight_left.push_back(noseforredness[2]);
		cheekRight_left.push_back(noseforredness[1]);
		//cheekRight_left.push_back(cheekLeft[0]);
		cheekRight_left.push_back(cheekLeft[1]);
		cheekRight_left.push_back(cheekLeft[2]);
		cheekRight_left.push_back(cv::Point(nasolabialLeft[7].x + (int)(0.5*(nasolabialLeft[8].x - nasolabialLeft[7].x)), nasolabialLeft[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));
		cheekRight_left.push_back(nasolabialLeft[0]);

		std::vector<cv::Point> cheekLeft_right, cheekLeft_left;
		for (int k = 2; k <= 9; k++)
		{
			cheekLeft_left.push_back(cheekRight[k]);
		}

		cheekLeft_left.push_back(nasolabialRight[5]);
		cheekLeft_left.push_back(nasolabialRight[6]);
		cheekLeft_left.push_back(cv::Point(nasolabialRight[7].x + (int)(0.5*(nasolabialRight[8].x - nasolabialRight[7].x)), nasolabialRight[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));

		//cheekLeft_right.push_back(cheekRight[12]);
		cheekLeft_right.push_back(cheekRight[13]);
		cheekLeft_right.push_back(noseforredness[11]);
		cheekLeft_right.push_back(noseforredness[12]);
		cheekLeft_right.push_back(noseforredness[13]);
		//cheekLeft_right.push_back(cheekRight[0]);
		cheekLeft_right.push_back(cheekRight[1]);
		cheekLeft_right.push_back(cheekRight[2]);
		cheekLeft_right.push_back(cv::Point(nasolabialRight[7].x + (int)(0.5*(nasolabialRight[8].x - nasolabialRight[7].x)), nasolabialRight[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));
		cheekLeft_right.push_back(nasolabialRight[0]);

		std::vector<cv::Point> Chin_Derma;
		for (int k = 7; k <= 11; k++)
		{
			Chin_Derma.push_back(chin[k]);
		}
		Chin_Derma.push_back(nasolabialLeft[4]);
		for (int k = 1; k <= 5; k++)
		{
			Chin_Derma.push_back(chin[k]);
		}
		Chin_Derma.push_back(nasolabialRight[4]);

		for (int k = 0; k < RightForeHead.size(); k++)
		{
			DermaRegions[0].second.push_back(RightForeHead[k]);
		}
		DermaRegions[0].first = "RightForeHead";

		for (int k = 0; k < MiddleForehead.size(); k++)
		{
			DermaRegions[1].second.push_back(MiddleForehead[k]);
		}
		DermaRegions[1].first = "MiddleForehead";

		for (int k = 0; k < LeftForeHead.size(); k++)
		{
			DermaRegions[2].second.push_back(LeftForeHead[k]);
		}
		DermaRegions[2].first = "LeftForeHead";

		for (int k = 0; k < glabella.size(); k++)
		{
			DermaRegions[3].second.push_back(glabella[k]);
		}
		DermaRegions[3].first = "Glabella";

		for (int k = 0; k < Nose_Derma.size(); k++)
		{
			DermaRegions[4].second.push_back(Nose_Derma[k]);
		}
		DermaRegions[4].first = "Nose";

		for (int k = 0; k < cheekRight_right.size(); k++)
		{
			DermaRegions[5].second.push_back(cheekRight_right[k]);
		}
		DermaRegions[5].first = "RightCheek";

		for (int k = 0; k < cheekRight_left.size(); k++)
		{
			DermaRegions[6].second.push_back(cheekRight_left[k]);
		}
		DermaRegions[6].first = "RightCheekCynus";

		for (int k = 0; k < cheekLeft_right.size(); k++)
		{
			DermaRegions[7].second.push_back(cheekLeft_right[k]);
		}
		DermaRegions[7].first = "CheekLeft";

		for (int k = 0; k < cheekLeft_left.size(); k++)
		{
			DermaRegions[8].second.push_back(cheekLeft_left[k]);
		}
		DermaRegions[8].first = "CheekLeftCynus";

		for (int k = 0; k < nasolabialRight.size() - 2; k++)
		{
			DermaRegions[9].second.push_back(nasolabialLeft[k]);
		}
		DermaRegions[9].second.push_back(cv::Point(nasolabialLeft[7].x + (int)(0.5*(nasolabialLeft[8].x - nasolabialLeft[7].x)), nasolabialLeft[7].y + (int)(0.5*(nasolabialLeft[8].y - nasolabialLeft[7].y))));

		DermaRegions[9].first = "NasolabialLeft";

		for (int k = 0; k < nasolabialRight.size() - 2; k++)
		{
			DermaRegions[10].second.push_back(nasolabialRight[k]);
		}
		DermaRegions[10].second.push_back(cv::Point(nasolabialRight[7].x + (int)(0.5*(nasolabialRight[8].x - nasolabialRight[7].x)), nasolabialRight[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));

		DermaRegions[10].first = "NasolabialRight";

		for (int k = 0; k < chin.size(); k++)
		{
			DermaRegions[11].second.push_back(Chin_Derma[k]);
		}
		DermaRegions[11].first = "Chin";
	}
	catch (const std::exception&ex)
	{
		std::cout << "Exception in draw 12 regions method " << ex.what() << std::endl;
	}
	return DermaRegions;
}

std::vector<std::pair<std::string, std::vector<cv::Point>>> Skincare::FacialRegions_Live::facialZones(std::vector<cv::Point> fldPoints, int no_of_regions)
{
	std::vector<std::pair<std::string, std::vector<cv::Point>>> DermaRegions;
	DermaRegions.resize(no_of_regions);

	try
	{
		const int foreheadLength = 15, glabellaLength = 4, noseLength = 7, underEyeLength = 8, cheekLength = 15, upperLipLength = 12, lipLength = 12, chinLength = 12;
		const int nasolabialLength = 9;

		std::vector<cv::Point> forehead(foreheadLength, cv::Point(0, 0));
		std::vector<cv::Point> glabella(glabellaLength, cv::Point(0, 0));
		std::vector<cv::Point> nose(noseLength, cv::Point(0, 0));
		std::vector<cv::Point> underEyeLeft(underEyeLength, cv::Point(0, 0));
		std::vector<cv::Point> underEyeRight(underEyeLength, cv::Point(0, 0));
		std::vector<cv::Point> cheekLeft(cheekLength, cv::Point(0, 0));
		std::vector<cv::Point> cheekRight(cheekLength, cv::Point(0, 0));
		std::vector<cv::Point> upperLip(lipLength, cv::Point(0, 0));
		std::vector<cv::Point> lip(chinLength, cv::Point(0, 0));
		std::vector<cv::Point> chin(chinLength, cv::Point(0, 0));
		std::vector<cv::Point> nasolabialLeft(nasolabialLength, cv::Point(0, 0));
		std::vector<cv::Point> nasolabialRight(nasolabialLength, cv::Point(0, 0));

		//region Forehead
		int YThresh_forehead = (int)((std::max(fldPoints[19].y, fldPoints[24].y) - fldPoints[71].y)*0.05);//Its always a +ve value
		int xThresh_glabella = (int)((fldPoints[22].x - fldPoints[21].x)*0.25);//Its always a +ve value
		double forheadverticalslope = 0;
		if ((fldPoints[8].x - fldPoints[27].x) != 0)
		{
			forheadverticalslope = (fldPoints[8].y - fldPoints[27].y) / (double)(fldPoints[8].x - fldPoints[27].x);
		}
		forehead[0].x = fldPoints[68].x; forehead[0].y = fldPoints[68].y + YThresh_forehead;
		forehead[1].x = fldPoints[69].x; forehead[1].y = fldPoints[69].y + YThresh_forehead;
		forehead[2].x = fldPoints[70].x; forehead[2].y = fldPoints[70].y + YThresh_forehead;
		forehead[3].x = fldPoints[71].x; forehead[3].y = fldPoints[71].y + YThresh_forehead;
		forehead[4].x = fldPoints[72].x; forehead[4].y = fldPoints[72].y + YThresh_forehead;
		forehead[5].x = fldPoints[73].x; forehead[5].y = fldPoints[73].y + YThresh_forehead;

		forehead[6].x = fldPoints[74].x; forehead[6].y = fldPoints[74].y + YThresh_forehead;
		forehead[8].x = fldPoints[24].x; forehead[8].y = fldPoints[24].y - YThresh_forehead;
		forehead[7].x = (int)(0.5*(fldPoints[25].x + forehead[6].x)); forehead[7].y = (int)(0.5*(forehead[6].y + forehead[8].y));

		forehead[9].x = fldPoints[23].x; forehead[9].y = fldPoints[23].y - YThresh_forehead;
		forehead[12].x = fldPoints[20].x; forehead[12].y = fldPoints[20].y - YThresh_forehead;
		forehead[13].x = fldPoints[19].x; forehead[13].y = fldPoints[19].y - YThresh_forehead;
		forehead[14].x = (int)(0.5*(fldPoints[18].x + forehead[0].x)); forehead[14].y = (int)(0.5*(forehead[0].y + forehead[12].y));

		double ForeheadSlope = (double)(forehead[9].y - forehead[12].y) / (forehead[9].x - forehead[12].x);//no need to check denominator is zero,its always give  a positive value

		cv::Point centerpoint(0, 0);
		centerpoint.x = (forehead[9].x + forehead[12].x) / 2;
		centerpoint.y = (forehead[9].y + forehead[12].y) / 2;

		forehead[10].x = centerpoint.x + xThresh_glabella * 2;//extra curvature points to connect with glabella
		forehead[10].y = forehead[12].y + (int)(ForeheadSlope*(forehead[10].x - forehead[12].x));

		forehead[11].x = centerpoint.x - xThresh_glabella * 2;
		forehead[11].y = forehead[12].y + (int)(ForeheadSlope*(forehead[11].x - forehead[12].x));

		//endregion

		//region Glabella
		glabella[0].x = forehead[11].x; glabella[0].y = forehead[11].y;
		glabella[1].x = forehead[10].x;    glabella[1].y = forehead[10].y;

		glabella[2].x = fldPoints[27].x + xThresh_glabella;//glabella points limiting upto nose start point
		glabella[2].y = fldPoints[27].y + (int)(ForeheadSlope*(glabella[2].x - fldPoints[27].x));

		glabella[3].x = fldPoints[27].x - xThresh_glabella;
		glabella[3].y = fldPoints[27].y + (int)(ForeheadSlope*(glabella[3].x - fldPoints[27].x));

		//endregion

		//region Nose
		int XThresh_noseLeft1 = (fldPoints[28].x - fldPoints[39].x);//To avoid offset when person in rotation,used different offsets for each nose point
		int XThresh_noseRight1 = (fldPoints[42].x - fldPoints[28].x);
		int XThresh_noseLeft2 = (fldPoints[29].x - fldPoints[39].x);
		int XThresh_noseRight2 = (fldPoints[42].x - fldPoints[29].x);
		int YThresh_nose = (fldPoints[30].y - fldPoints[29].y);//To identify nose edge

		nose[0].x = glabella[3].x; nose[0].y = glabella[3].y;
		nose[6].x = glabella[2].x; nose[6].y = glabella[2].y;

		nose[1].x = fldPoints[28].x - (int)((0.6*XThresh_noseLeft1));
		nose[1].y = fldPoints[28].y + (int)(ForeheadSlope*(nose[1].x - fldPoints[28].x));

		nose[5].x = fldPoints[28].x + (int)((0.6*XThresh_noseRight1));
		nose[5].y = fldPoints[28].y + (int)(ForeheadSlope*(nose[5].x - fldPoints[28].x));

		nose[2].x = fldPoints[29].x - (int)((0.75*XThresh_noseLeft2));
		nose[2].y = fldPoints[29].y + (int)(ForeheadSlope*(nose[2].x - fldPoints[29].x));

		nose[4].x = fldPoints[29].x + (int)((0.75*XThresh_noseRight2));
		nose[4].y = fldPoints[29].y + (int)(ForeheadSlope*(nose[4].x - fldPoints[29].x));

		nose[3].x = fldPoints[29].x;
		nose[3].y = fldPoints[29].y + (int)(YThresh_nose*0.7);
		//endregion

		//region Undereyes
		int YThresh_underEye = (int)((fldPoints[8].y - fldPoints[27].y)*0.15);//Its always a +ve value
		int XThresh_underEyeRight = (int)((fldPoints[39].x - fldPoints[36].x)*0.1);//Its always a +ve value

		underEyeRight[1].x = fldPoints[40].x;
		underEyeRight[1].y = fldPoints[40].y + (int)(0.25 * YThresh_underEye);

		underEyeRight[2].x = fldPoints[41].x;
		underEyeRight[2].y = fldPoints[41].y + (int)(0.25 * YThresh_underEye);

		underEyeRight[5].x = underEyeRight[2].x;
		underEyeRight[5].y = underEyeRight[2].y + (int)(0.8*YThresh_underEye);

		underEyeRight[6].x = underEyeRight[1].x;
		underEyeRight[6].y = underEyeRight[1].y + (int)(0.8*YThresh_underEye);

		underEyeRight[0].x = fldPoints[39].x;
		underEyeRight[0].y = fldPoints[39].y + (int)(0.15 * YThresh_underEye);

		underEyeRight[3].x = (fldPoints[36].x + fldPoints[0].x) / 2;
		underEyeRight[3].y = fldPoints[36].y + (int)(0.05 * YThresh_underEye);

		underEyeRight[4].x = underEyeRight[3].x - (2 * XThresh_underEyeRight);
		underEyeRight[4].y = underEyeRight[3].y + (int)(0.6*(underEyeRight[5].y - underEyeRight[3].y));

		underEyeRight[7].x = underEyeRight[0].x + XThresh_underEyeRight;
		underEyeRight[7].y = underEyeRight[0].y + (int)(0.6*(underEyeRight[6].y - underEyeRight[0].y));

		//underEyeRight
		int XThresh_underEyeLeft = (int)((fldPoints[45].x - fldPoints[42].x)*0.1);

		underEyeLeft[1].x = fldPoints[47].x;
		underEyeLeft[1].y = fldPoints[47].y + (int)(0.25 * YThresh_underEye);

		underEyeLeft[2].x = fldPoints[46].x;
		underEyeLeft[2].y = fldPoints[46].y + (int)(0.25 * YThresh_underEye);

		underEyeLeft[5].x = underEyeLeft[2].x;
		underEyeLeft[5].y = underEyeLeft[2].y + (int)(0.8*YThresh_underEye);

		underEyeLeft[6].x = underEyeLeft[1].x;
		underEyeLeft[6].y = underEyeLeft[1].y + (int)(0.8*YThresh_underEye);

		underEyeLeft[0].x = fldPoints[42].x;
		underEyeLeft[0].y = fldPoints[42].y + (int)(0.15 * YThresh_underEye);

		underEyeLeft[3].x = (fldPoints[45].x + fldPoints[16].x) / 2;
		underEyeLeft[3].y = fldPoints[45].y + (int)(0.05 * YThresh_underEye);

		underEyeLeft[4].x = underEyeLeft[3].x + (2 * XThresh_underEyeLeft);
		underEyeLeft[4].y = underEyeLeft[3].y + (int)(0.6*(underEyeLeft[5].y - underEyeLeft[3].y));

		underEyeLeft[7].x = underEyeLeft[0].x - XThresh_underEyeLeft;
		underEyeLeft[7].y = underEyeLeft[0].y + (int)(0.6*(underEyeLeft[6].y - underEyeLeft[0].y));
		//endregion

		//region Upper lip
		int YThresh_upperLip = (int)((fldPoints[51].y - fldPoints[33].y)*0.1);
		upperLip[0].x = fldPoints[31].x; upperLip[0].y = fldPoints[31].y + YThresh_upperLip;
		upperLip[1].x = fldPoints[33].x; upperLip[1].y = fldPoints[33].y + YThresh_upperLip;
		upperLip[2].x = fldPoints[35].x; upperLip[2].y = fldPoints[35].y + YThresh_upperLip;
		upperLip[4].x = fldPoints[54].x; upperLip[4].y = fldPoints[54].y - YThresh_upperLip;
		upperLip[3].x = upperLip[4].x - (int)(0.3*(upperLip[4].x - upperLip[2].x)); upperLip[3].y = (upperLip[4].y + upperLip[2].y) / 2;//curvature point.
		upperLip[5].x = fldPoints[53].x; upperLip[5].y = fldPoints[53].y - YThresh_upperLip;
		upperLip[6].x = fldPoints[52].x; upperLip[6].y = fldPoints[52].y - YThresh_upperLip;
		upperLip[7].x = fldPoints[51].x; upperLip[7].y = fldPoints[51].y - YThresh_upperLip;
		upperLip[8].x = fldPoints[50].x; upperLip[8].y = fldPoints[50].y - YThresh_upperLip;
		upperLip[9].x = fldPoints[49].x; upperLip[9].y = fldPoints[49].y - YThresh_upperLip;
		upperLip[10].x = fldPoints[48].x; upperLip[10].y = fldPoints[48].y - YThresh_upperLip;
		upperLip[11].x = upperLip[10].x - (int)(0.3*(upperLip[10].x - upperLip[0].x)); upperLip[11].y = (upperLip[10].y + upperLip[0].y) / 2;//curvature point.
		//endregion
		//region Chin overlap
		int XThresh_chin = (int)((fldPoints[54].x - fldPoints[48].x)*0.1);//Its always a +ve value
		int YThresh_chin = (int)((fldPoints[8].y - fldPoints[57].y)*0.1);//Its always a +ve value
		int YThresh_chin1 = (int)((fldPoints[8].y - fldPoints[48].y)*0.1); //Its always a +ve value
		int YThresh_chin2 = (int)((fldPoints[8].y - fldPoints[54].y)*0.1);//Its always a +ve value

		chin[0].x = fldPoints[48].x - XThresh_chin; chin[0].y = fldPoints[48].y + YThresh_chin1;
		chin[1].x = fldPoints[59].x; chin[1].y = fldPoints[59].y + YThresh_chin;//these are down chin points,so used different offset for above and down points
		chin[2].x = fldPoints[58].x; chin[2].y = fldPoints[58].y + YThresh_chin;
		chin[3].x = fldPoints[57].x; chin[3].y = fldPoints[57].y + YThresh_chin;
		chin[4].x = fldPoints[56].x; chin[4].y = fldPoints[56].y + YThresh_chin;
		chin[5].x = fldPoints[55].x; chin[5].y = fldPoints[55].y + YThresh_chin;


		chin[6].x = fldPoints[54].x + XThresh_chin; chin[6].y = fldPoints[54].y + YThresh_chin2;//To avoid offset variations for straight and tilt faces,used different offsets for left & right points
		chin[7].x = fldPoints[10].x; chin[7].y = fldPoints[10].y - YThresh_chin;
		chin[8].x = fldPoints[9].x; chin[8].y = fldPoints[9].y - YThresh_chin;
		chin[9].x = fldPoints[8].x; chin[9].y = fldPoints[8].y - YThresh_chin;
		chin[10].x = fldPoints[7].x; chin[10].y = fldPoints[7].y - YThresh_chin;
		chin[11].x = fldPoints[6].x; chin[11].y = fldPoints[6].y - YThresh_chin;
		//endregion

		//endregion

		//region Cheeks
		//cheekLeft
		int XThresh_cheekRight = (int)((fldPoints[30].x - fldPoints[2].x)*0.05);
		int XThresh_noseEdgeRight = fldPoints[33].x - fldPoints[31].x;
		cheekRight[0].x = underEyeRight[7].x; cheekRight[0].y = underEyeRight[7].y;
		cheekRight[1].x = underEyeRight[6].x; cheekRight[1].y = underEyeRight[6].y;
		cheekRight[2].x = underEyeRight[5].x; cheekRight[2].y = underEyeRight[5].y;
		cheekRight[3].x = underEyeRight[4].x; cheekRight[3].y = underEyeRight[4].y;
		cheekRight[4].x = fldPoints[2].x + XThresh_cheekRight; cheekRight[4].y = fldPoints[2].y;
		cheekRight[5].x = fldPoints[3].x + XThresh_cheekRight; cheekRight[5].y = fldPoints[3].y;
		cheekRight[6].x = fldPoints[4].x + XThresh_cheekRight; cheekRight[6].y = fldPoints[4].y;
		cheekRight[7].x = fldPoints[5].x + XThresh_cheekRight; cheekRight[7].y = fldPoints[5].y;

		cheekRight[8].x = chin[11].x; cheekRight[8].y = chin[11].y;
		cheekRight[10].x = chin[1].x; cheekRight[10].y = chin[1].y;
		cheekRight[9].x = cheekRight[8].x - (int)(0.3*(cheekRight[8].x - cheekRight[10].x)); cheekRight[9].y = (cheekRight[8].y + cheekRight[10].y) / 2;


		cheekRight[11].x = fldPoints[48].x - XThresh_cheekRight; cheekRight[11].y = fldPoints[48].y;
		cheekRight[13].x = fldPoints[31].x - (int)(0.5*XThresh_noseEdgeRight); cheekRight[13].y = fldPoints[31].y;
		cheekRight[12].x = upperLip[11].x; cheekRight[12].y = upperLip[11].y;
		cheekRight[14].x = fldPoints[31].x - (int)(0.6*XThresh_noseEdgeRight); cheekRight[14].y = (int)(0.5*(cheekRight[13].y + cheekRight[0].y));

		//cheekRight
		int XThresh_cheekLeft = (int)((fldPoints[14].x - fldPoints[30].x)*0.05);
		int XThresh_noseEdgeLeft = fldPoints[35].x - fldPoints[33].x;
		cheekLeft[0].x = underEyeLeft[7].x; cheekLeft[0].y = underEyeLeft[7].y;
		cheekLeft[1].x = underEyeLeft[6].x; cheekLeft[1].y = underEyeLeft[6].y;
		cheekLeft[2].x = underEyeLeft[5].x; cheekLeft[2].y = underEyeLeft[5].y;
		cheekLeft[3].x = underEyeLeft[4].x; cheekLeft[3].y = underEyeLeft[4].y;
		cheekLeft[4].x = fldPoints[14].x - XThresh_cheekLeft; cheekLeft[4].y = fldPoints[14].y;
		cheekLeft[5].x = fldPoints[13].x - XThresh_cheekLeft; cheekLeft[5].y = fldPoints[13].y;
		cheekLeft[6].x = fldPoints[12].x - XThresh_cheekLeft; cheekLeft[6].y = fldPoints[12].y;
		cheekLeft[7].x = fldPoints[11].x - XThresh_cheekLeft; cheekLeft[7].y = fldPoints[11].y;

		cheekLeft[8].x = chin[7].x; cheekLeft[8].y = chin[7].y;
		cheekLeft[10].x = chin[5].x; cheekLeft[10].y = chin[5].y;
		cheekLeft[9].x = cheekLeft[8].x - (int)(0.3*(cheekLeft[8].x - cheekLeft[10].x)); cheekLeft[9].y = (cheekLeft[8].y + cheekLeft[10].y) / 2;

		cheekLeft[11].x = fldPoints[54].x + XThresh_cheekLeft; cheekLeft[11].y = fldPoints[54].y;
		cheekLeft[13].x = fldPoints[35].x + (int)(0.5*XThresh_noseEdgeLeft); cheekLeft[13].y = fldPoints[35].y;
		cheekLeft[12].x = upperLip[3].x; cheekLeft[12].y = upperLip[3].y;
		cheekLeft[14].x = fldPoints[35].x + (int)(0.6*XThresh_noseEdgeLeft); cheekLeft[14].y = (cheekLeft[13].y + cheekLeft[0].y) / 2;
		//endregion

		//region Lip
		lip[0].x = fldPoints[48].x; lip[0].y = fldPoints[48].y;
		lip[1].x = fldPoints[49].x; lip[1].y = fldPoints[49].y;
		lip[2].x = fldPoints[50].x; lip[2].y = fldPoints[50].y;
		lip[3].x = fldPoints[51].x; lip[3].y = fldPoints[51].y;
		lip[4].x = fldPoints[52].x; lip[4].y = fldPoints[52].y;
		lip[5].x = fldPoints[53].x; lip[5].y = fldPoints[53].y;
		lip[6].x = fldPoints[54].x; lip[6].y = fldPoints[54].y;
		lip[7].x = fldPoints[55].x; lip[7].y = fldPoints[55].y;
		lip[8].x = fldPoints[56].x; lip[8].y = fldPoints[56].y;
		lip[9].x = fldPoints[57].x; lip[9].y = fldPoints[57].y;
		lip[10].x = fldPoints[58].x; lip[10].y = fldPoints[58].y;
		lip[11].x = fldPoints[59].x; lip[11].y = fldPoints[59].y;
		//endregion

		//region Nasolabial
		//Nasolabial Left
		int XThresh_nasolabialRight1 = (fldPoints[31].x - cheekRight[5].x);//Its always a +ve value
		int XThresh_nasolabialRight2 = (fldPoints[48].x - cheekRight[6].x);//Its always a +ve value
		int XThresh_nasolabialRight3 = (fldPoints[59].x - cheekRight[7].x);//Its always a +ve value
		int YThresh_nasolabial = (int)((fldPoints[51].y - fldPoints[33].y)*0.3);//Its always a +ve value

		nasolabialRight[0].x = cheekRight[13].x; nasolabialRight[0].y = cheekRight[13].y;
		nasolabialRight[1].x = cheekRight[12].x; nasolabialRight[1].y = cheekRight[12].y;
		nasolabialRight[2].x = cheekRight[11].x; nasolabialRight[2].y = cheekRight[11].y;
		nasolabialRight[3].x = cheekRight[10].x; nasolabialRight[3].y = cheekRight[10].y;
		nasolabialRight[4].x = cheekRight[9].x; nasolabialRight[4].y = cheekRight[9].y;
		nasolabialRight[5].x = nasolabialRight[3].x - (int)((XThresh_nasolabialRight3*0.8)); nasolabialRight[5].y = nasolabialRight[3].y;
		nasolabialRight[6].x = nasolabialRight[2].x - (int)((XThresh_nasolabialRight2*0.5)); nasolabialRight[6].y = nasolabialRight[2].y;
		nasolabialRight[7].x = nasolabialRight[1].x - (int)((XThresh_nasolabialRight1*0.4)); nasolabialRight[7].y = nasolabialRight[1].y;
		nasolabialRight[8].x = nasolabialRight[1].x - ((int)(XThresh_nasolabialRight1*0.2)); nasolabialRight[8].y = nasolabialRight[0].y + YThresh_nasolabial;

		//Nasolabial Right
		int XThresh_nasolabialLeft1 = (cheekLeft[5].x - fldPoints[35].x);//Its always a +ve value
		int XThresh_nasolabialLeft2 = (cheekLeft[6].x - fldPoints[54].x);//Its always a +ve value
		int XThresh_nasolabialLeft3 = (cheekLeft[7].x - fldPoints[55].x);//Its always a +ve value

		nasolabialLeft[0].x = cheekLeft[13].x; nasolabialLeft[0].y = cheekLeft[13].y;
		nasolabialLeft[1].x = cheekLeft[12].x; nasolabialLeft[1].y = cheekLeft[12].y;
		nasolabialLeft[2].x = cheekLeft[11].x; nasolabialLeft[2].y = cheekLeft[11].y;
		nasolabialLeft[3].x = cheekLeft[10].x; nasolabialLeft[3].y = cheekLeft[10].y;

		nasolabialLeft[4].x = cheekLeft[9].x; nasolabialLeft[4].y = cheekLeft[9].y;
		nasolabialLeft[5].x = nasolabialLeft[3].x + (int)((XThresh_nasolabialLeft3*0.8)); nasolabialLeft[5].y = nasolabialLeft[3].y;
		nasolabialLeft[6].x = nasolabialLeft[2].x + (int)((XThresh_nasolabialLeft2*0.5)); nasolabialLeft[6].y = nasolabialLeft[2].y;
		nasolabialLeft[7].x = nasolabialLeft[1].x + (int)((XThresh_nasolabialLeft1*0.4)); nasolabialLeft[7].y = nasolabialLeft[1].y;
		nasolabialLeft[8].x = nasolabialLeft[1].x + (int)((XThresh_nasolabialLeft1*0.2)); nasolabialLeft[8].y = nasolabialLeft[0].y + YThresh_nasolabial;
		// endregion

		//region noseforredness
		const int noseforrednessLength = 15;
		std::vector<cv::Point> noseforredness(noseforrednessLength, cv::Point(0, 0));

		int XThresh_noseLeft3 = (fldPoints[30].x - fldPoints[39].x);//Its always a +ve value
		int XThresh_noseRight3 = (fldPoints[42].x - fldPoints[30].x);//Its always a +ve value

		int YThresh_nose1 = (int)(0.4*(fldPoints[51].y - fldPoints[33].y));

		noseforredness[0].x = nose[0].x; noseforredness[0].y = nose[0].y;
		noseforredness[1].x = nose[1].x; noseforredness[1].y = nose[1].y;
		noseforredness[2].x = nose[2].x; noseforredness[2].y = nose[2].y;

		noseforredness[12].x = nose[4].x; noseforredness[12].y = nose[4].y;
		noseforredness[13].x = nose[5].x; noseforredness[13].y = nose[5].y;
		noseforredness[14].x = nose[6].x; noseforredness[14].y = nose[6].y;

		noseforredness[3].x = fldPoints[30].x - (int)(1 * XThresh_noseLeft3);
		noseforredness[3].y = fldPoints[30].y + (int)(ForeheadSlope*(noseforredness[3].x - fldPoints[30].x));

		noseforredness[4].x = cheekRight[13].x; noseforredness[4].y = cheekRight[13].y;
		noseforredness[10].x = cheekLeft[13].x; noseforredness[10].y = cheekLeft[13].y;

		noseforredness[11].x = fldPoints[30].x + (int)(1 * XThresh_noseRight3);
		noseforredness[11].y = fldPoints[30].y + (int)(ForeheadSlope*(noseforredness[11].x - fldPoints[30].x));

		noseforredness[5].x = fldPoints[31].x;
		noseforredness[5].y = fldPoints[31].y + YThresh_nose1;

		noseforredness[6].x = fldPoints[32].x;
		noseforredness[6].y = fldPoints[32].y + YThresh_nose1;

		noseforredness[7].x = fldPoints[33].x;
		noseforredness[7].y = fldPoints[33].y + YThresh_nose1;

		noseforredness[8].x = fldPoints[34].x;
		noseforredness[8].y = fldPoints[34].y + YThresh_nose1;

		noseforredness[9].x = fldPoints[35].x;
		noseforredness[9].y = fldPoints[35].y + YThresh_nose1;
		//endregion

		//region cheekforredness
		const int cheekforrednessLength = 16;
		std::vector<cv::Point> cheekLeftforredness(cheekforrednessLength, cv::Point(0, 0));
		std::vector<cv::Point> cheekRightforredness(cheekforrednessLength, cv::Point(0, 0));

		//Combining cheek redness points with nose redness region end points
		cheekLeftforredness[0].x = cheekLeft[1].x; cheekLeftforredness[0].y = cheekLeft[1].y;
		cheekLeftforredness[1].x = cheekLeft[2].x; cheekLeftforredness[1].y = cheekLeft[2].y;
		cheekLeftforredness[2].x = cheekLeft[3].x; cheekLeftforredness[2].y = cheekLeft[3].y;
		cheekLeftforredness[3].x = cheekLeft[4].x; cheekLeftforredness[3].y = cheekLeft[4].y;
		cheekLeftforredness[4].x = cheekLeft[5].x; cheekLeftforredness[4].y = cheekLeft[5].y;
		cheekLeftforredness[5].x = cheekLeft[6].x; cheekLeftforredness[5].y = cheekLeft[6].y;
		cheekLeftforredness[6].x = cheekLeft[7].x; cheekLeftforredness[6].y = cheekLeft[7].y;
		cheekLeftforredness[7].x = cheekLeft[8].x; cheekLeftforredness[7].y = cheekLeft[8].y;
		cheekLeftforredness[8].x = cheekLeft[9].x; cheekLeftforredness[8].y = cheekLeft[9].y;
		cheekLeftforredness[9].x = cheekLeft[10].x; cheekLeftforredness[9].y = cheekLeft[10].y;
		cheekLeftforredness[10].x = cheekLeft[11].x; cheekLeftforredness[10].y = cheekLeft[11].y;
		cheekLeftforredness[11].x = cheekLeft[12].x; cheekLeftforredness[11].y = cheekLeft[12].y;
		cheekLeftforredness[12].x = cheekLeft[13].x; cheekLeftforredness[12].y = cheekLeft[13].y;
		cheekLeftforredness[13].x = noseforredness[3].x; cheekLeftforredness[13].y = noseforredness[3].y;
		cheekLeftforredness[14].x = noseforredness[2].x; cheekLeftforredness[14].y = noseforredness[2].y;
		cheekLeftforredness[15].x = noseforredness[1].x; cheekLeftforredness[15].y = noseforredness[1].y;

		cheekRightforredness[0].x = cheekRight[1].x; cheekRightforredness[0].y = cheekRight[1].y;
		cheekRightforredness[1].x = cheekRight[2].x; cheekRightforredness[1].y = cheekRight[2].y;
		cheekRightforredness[2].x = cheekRight[3].x; cheekRightforredness[2].y = cheekRight[3].y;
		cheekRightforredness[3].x = cheekRight[4].x; cheekRightforredness[3].y = cheekRight[4].y;
		cheekRightforredness[4].x = cheekRight[5].x; cheekRightforredness[4].y = cheekRight[5].y;
		cheekRightforredness[5].x = cheekRight[6].x; cheekRightforredness[5].y = cheekRight[6].y;
		cheekRightforredness[6].x = cheekRight[7].x; cheekRightforredness[6].y = cheekRight[7].y;
		cheekRightforredness[7].x = cheekRight[8].x; cheekRightforredness[7].y = cheekRight[8].y;
		cheekRightforredness[8].x = cheekRight[9].x; cheekRightforredness[8].y = cheekRight[9].y;
		cheekRightforredness[9].x = cheekRight[10].x; cheekRightforredness[9].y = cheekRight[10].y;
		cheekRightforredness[10].x = cheekRight[11].x; cheekRightforredness[10].y = cheekRight[11].y;
		cheekRightforredness[11].x = cheekRight[12].x; cheekRightforredness[11].y = cheekRight[12].y;
		cheekRightforredness[12].x = cheekRight[13].x; cheekRightforredness[12].y = cheekRight[13].y;
		cheekRightforredness[13].x = noseforredness[11].x; cheekRightforredness[13].y = noseforredness[11].y;
		cheekRightforredness[14].x = noseforredness[12].x; cheekRightforredness[14].y = noseforredness[12].y;
		cheekRightforredness[15].x = noseforredness[13].x; cheekRightforredness[15].y = noseforredness[13].y;

		//endregion
		// region cheekforUEC
		const int cheekforUECLength = 6;
		std::vector<cv::Point> cheekLeftforUEC(cheekforUECLength, cv::Point(0, 0));
		std::vector<cv::Point> cheekRightforUEC(cheekforUECLength, cv::Point(0, 0));

		cheekLeftforUEC[0].x = cheekRight[0].x; cheekLeftforUEC[0].y = cheekRight[0].y;
		cheekLeftforUEC[1].x = cheekRight[1].x; cheekLeftforUEC[1].y = cheekRight[1].y;
		cheekLeftforUEC[2].x = cheekRight[2].x; cheekLeftforUEC[2].y = cheekRight[2].y;
		cheekLeftforUEC[3].x = cheekRight[3].x; cheekLeftforUEC[3].y = cheekRight[3].y;
		cheekLeftforUEC[5].x = cheekRight[13].x; cheekLeftforUEC[5].y = cheekRight[13].y;
		cheekLeftforUEC[4].x = cheekLeftforUEC[5].x - (int)((cheekLeftforUEC[5].x - cheekRight[5].x)*0.8);
		cheekLeftforUEC[4].y = cheekLeftforUEC[5].y + (int)(ForeheadSlope*(cheekLeftforUEC[4].x - cheekLeftforUEC[5].x));

		cheekRightforUEC[0].x = cheekLeft[0].x; cheekRightforUEC[0].y = cheekLeft[0].y;
		cheekRightforUEC[1].x = cheekLeft[1].x; cheekRightforUEC[1].y = cheekLeft[1].y;
		cheekRightforUEC[2].x = cheekLeft[2].x; cheekRightforUEC[2].y = cheekLeft[2].y;
		cheekRightforUEC[3].x = cheekLeft[3].x; cheekRightforUEC[3].y = cheekLeft[3].y;
		cheekRightforUEC[5].x = cheekLeft[13].x; cheekRightforUEC[5].y = cheekLeft[13].y;
		cheekRightforUEC[4].x = cheekRightforUEC[5].x + (int)((cheekLeft[5].x - cheekRightforUEC[5].x)*0.8);
		cheekRightforUEC[4].y = cheekRightforUEC[5].y + (int)(ForeheadSlope*(cheekRightforUEC[4].x - cheekRightforUEC[5].x));
		// endregion

		std::vector<cv::Point> Chin_Derma;
		for (int k = 7; k <= 11; k++)
		{
			Chin_Derma.push_back(chin[k]);
		}
		Chin_Derma.push_back(nasolabialRight[4]);
		for (int k = 1; k <= 5; k++)
		{
			Chin_Derma.push_back(chin[k]);
		}
		Chin_Derma.push_back(nasolabialLeft[4]);

		if (no_of_regions == 4)
		{
			for (int k = 0; k < forehead.size(); k++)
			{
				DermaRegions[0].second.push_back(forehead[k]);
			}
			DermaRegions[0].first = "ForeHead";
			for (int k = 0; k < cheekLeftforredness.size(); k++)
			{
				DermaRegions[1].second.push_back(cheekLeftforredness[k]);
			}
			DermaRegions[1].first = "CheekLeft";
			for (int k = 0; k < cheekRightforredness.size(); k++)
			{
				DermaRegions[2].second.push_back(cheekRightforredness[k]);
			}
			DermaRegions[2].first = "CheekRight";
			for (int k = 0; k < Chin_Derma.size(); k++)
			{
				DermaRegions[3].second.push_back(Chin_Derma[k]);
			}
			DermaRegions[3].first = "Chin";
		}
		if (no_of_regions == 5)
		{
			std::vector<cv::Point> dataPoints;
			for (int k = 0; k < 11; k++)
			{
				dataPoints.push_back(forehead[k]);
			}
			dataPoints.push_back(glabella[2]);
			dataPoints.push_back(glabella[3]);
			for (int k = 11; k < forehead.size(); k++)
			{
				dataPoints.push_back(forehead[k]);
			}
			dataPoints.push_back(dataPoints[0]);
			getExtendedPoints(dataPoints, DermaRegions[0].second);
			DermaRegions[0].first = "ForeHead+Glabella";
			
			dataPoints.clear();
			for (int k = 0; k <= 3; k++)
			{
				dataPoints.push_back(cheekLeft[k]);
			}
			getExtendedPoints(dataPoints, DermaRegions[1].second, std::vector<int>(), false);
			dataPoints.clear();
			for (int k = 3; k <= 6; k++)
			{
				dataPoints.push_back(cheekLeft[k]);
			}
			getExtendedPoints(dataPoints, DermaRegions[1].second);
			dataPoints.clear();
			dataPoints.push_back(cheekLeft[6]);
			for (int k = 7; k <= 8; k++)
			{
				dataPoints.push_back(nasolabialLeft[k]);
			}
			dataPoints.push_back(nasolabialLeft[0]);
			dataPoints.push_back(cv::Point(nose[4].x, fldPoints[30].y));
			getExtendedPoints(dataPoints, DermaRegions[1].second, std::vector<int>(), false);
			dataPoints.clear();
			DermaRegions[1].second.push_back(cv::Point(nose[4].x, fldPoints[30].y));
			for (int k = 30; k >= 28; k--)
			{
				DermaRegions[1].second.push_back(fldPoints[k]);
			}
			cv::Point glabellMidPt = cv::Point((glabella[2].x + glabella[3].x) / 2, (glabella[2].y + glabella[3].y) / 2);
			DermaRegions[1].second.push_back(glabellMidPt);
			dataPoints.push_back(glabella[2]);
			dataPoints.push_back(nose[5]);
			dataPoints.push_back(cheekLeft[0]);
			getExtendedPoints(dataPoints, DermaRegions[1].second, std::vector<int>(), false);
			DermaRegions[1].first = "CheekLeft";

			dataPoints.clear();
			for (int k = 0; k <= 3; k++)
			{
				dataPoints.push_back(cheekRight[k]);
			}
			getExtendedPoints(dataPoints, DermaRegions[2].second);
			dataPoints.clear();
			for (int k = 3; k <= 6; k++)
			{
				dataPoints.push_back(cheekRight[k]);
			}
			getExtendedPoints(dataPoints, DermaRegions[2].second, std::vector<int>(), false);
			dataPoints.clear();
			dataPoints.push_back(cheekRight[6]);
			for (int k = 7; k <= 8; k++)
			{
				dataPoints.push_back(nasolabialRight[k]);
			}
			dataPoints.push_back(nasolabialRight[0]);
			dataPoints.push_back(cv::Point(nose[2].x, fldPoints[30].y));
			getExtendedPoints(dataPoints, DermaRegions[2].second);
			dataPoints.clear();
			DermaRegions[2].second.push_back(cv::Point(nose[2].x, fldPoints[30].y));
			for (int k = 30; k >= 28; k--)
			{
				DermaRegions[2].second.push_back(fldPoints[k]);
			}
			DermaRegions[2].second.push_back(glabellMidPt);
			dataPoints.push_back(glabella[3]);
			dataPoints.push_back(nose[1]);
			dataPoints.push_back(cheekRight[0]);
			getExtendedPoints(dataPoints, DermaRegions[2].second);
			DermaRegions[2].first = "CheekRight";

			dataPoints.clear();
			for (int k = 31; k <= 35; k++)
			{
				dataPoints.push_back(fldPoints[k]);
			}
			dataPoints.push_back(nasolabialLeft[0]);
			for (int k = 8; k >= 7; k--)
			{
				dataPoints.push_back(nasolabialLeft[k]);
			}
			for (int k = 6; k <= 8; k++)
			{
				dataPoints.push_back(cheekLeft[k]);
			}
			for (int k = 7; k <= 11; k++)
			{
				dataPoints.push_back(chin[k]);
			}

			for (int k = 8; k >= 6; k--)
			{
				dataPoints.push_back(cheekRight[k]);
			}
			dataPoints.push_back(nasolabialRight[7]);
			dataPoints.push_back(nasolabialRight[8]);
			dataPoints.push_back(nasolabialRight[0]);
			dataPoints.push_back(dataPoints[0]);
			getExtendedPoints(dataPoints, DermaRegions[3].second);
			DermaRegions[3].first = "Nasolabial";

			dataPoints.clear();
			for (int k = 48; k < 59; k++)
			{
				dataPoints.push_back(fldPoints[k]);
			}
			dataPoints.push_back(dataPoints[0]);
			getExtendedPoints(dataPoints, DermaRegions[4].second);
			DermaRegions[4].first = "Mouth";

		}
		else if (no_of_regions == 12)
		{
			std::vector<cv::Point> RightForeHead, LeftForeHead, MiddleForehead;
			RightForeHead.push_back(forehead[0]);
			RightForeHead.push_back(forehead[1]);
			RightForeHead.push_back(forehead[2]);
			RightForeHead.push_back(forehead[12]);
			RightForeHead.push_back(forehead[13]);
			RightForeHead.push_back(forehead[14]);

			MiddleForehead.push_back(forehead[2]);
			MiddleForehead.push_back(forehead[3]);
			MiddleForehead.push_back(forehead[4]);
			MiddleForehead.push_back(forehead[9]);
			MiddleForehead.push_back(forehead[10]);
			MiddleForehead.push_back(forehead[11]);
			MiddleForehead.push_back(forehead[12]);

			LeftForeHead.push_back(forehead[4]);
			LeftForeHead.push_back(forehead[5]);
			LeftForeHead.push_back(forehead[6]);
			LeftForeHead.push_back(forehead[7]);
			LeftForeHead.push_back(forehead[8]);
			LeftForeHead.push_back(forehead[9]);

			std::vector<cv::Point> Nose_Derma;
			Nose_Derma.push_back(noseforredness[0]);
			Nose_Derma.push_back(noseforredness[1]);
			Nose_Derma.push_back(noseforredness[2]);
			Nose_Derma.push_back(noseforredness[3]);
			Nose_Derma.push_back(cv::Point(fldPoints[30].x, fldPoints[30].y + (int)(0.5*abs(fldPoints[33].y - fldPoints[30].y))));
			Nose_Derma.push_back(noseforredness[11]);
			Nose_Derma.push_back(noseforredness[12]);
			Nose_Derma.push_back(noseforredness[13]);
			Nose_Derma.push_back(noseforredness[14]);

			std::vector<cv::Point> left_cheek_mouth, left_cheek_inner;

			for (int k = 2; k <= 9; k++)
			{
				left_cheek_mouth.push_back(cheekLeft[k]);
			}
			left_cheek_mouth.push_back(nasolabialLeft[5]);
			left_cheek_mouth.push_back(nasolabialLeft[6]);
			left_cheek_mouth.push_back(cv::Point(nasolabialLeft[7].x + (int)(0.5*(nasolabialLeft[8].x - nasolabialLeft[7].x)), nasolabialLeft[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));

			//cheekRight_left.push_back(cheekLeft[12]);
			left_cheek_inner.push_back(cheekLeft[13]);
			left_cheek_inner.push_back(noseforredness[11]);
			left_cheek_inner.push_back(noseforredness[12]);
			left_cheek_inner.push_back(noseforredness[13]);
			//cheekRight_left.push_back(cheekLeft[0]);
			left_cheek_inner.push_back(cheekLeft[1]);
			left_cheek_inner.push_back(cheekLeft[2]);
			left_cheek_inner.push_back(cv::Point(nasolabialLeft[7].x + (int)(0.5*(nasolabialLeft[8].x - nasolabialLeft[7].x)), nasolabialLeft[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));
			left_cheek_inner.push_back(nasolabialLeft[0]);

			std::vector<cv::Point> right_cheek_inner, right_cheek_mouth;
			for (int k = 2; k <= 9; k++)
			{
				right_cheek_mouth.push_back(cheekRight[k]);
			}

			right_cheek_mouth.push_back(nasolabialRight[5]);
			right_cheek_mouth.push_back(nasolabialRight[6]);
			right_cheek_mouth.push_back(cv::Point(nasolabialRight[7].x + (int)(0.5*(nasolabialRight[8].x - nasolabialRight[7].x)), nasolabialRight[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));

			//cheekLeft_right.push_back(cheekRight[12]);
			right_cheek_inner.push_back(cheekRight[13]);
			right_cheek_inner.push_back(noseforredness[3]);
			right_cheek_inner.push_back(noseforredness[2]);
			right_cheek_inner.push_back(noseforredness[1]);
			//cheekLeft_right.push_back(cheekRight[0]);
			right_cheek_inner.push_back(cheekRight[1]);
			right_cheek_inner.push_back(cheekRight[2]);
			right_cheek_inner.push_back(cv::Point(nasolabialRight[7].x + (int)(0.5*(nasolabialRight[8].x - nasolabialRight[7].x)), nasolabialRight[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));
			right_cheek_inner.push_back(nasolabialRight[0]);

			std::vector<cv::Point> dataPoints;
			for (int k = 0; k < RightForeHead.size(); k++)
			{
				dataPoints.push_back(RightForeHead[k]);
			}
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[0].first = "RightForeHead";
			std::vector<int> skipPoints = std::vector<int>{ 2 };
			getExtendedPoints(dataPoints, DermaRegions[0].second, skipPoints);

			dataPoints.clear();
			for (int k = 0; k < MiddleForehead.size(); k++)
			{
				dataPoints.push_back(MiddleForehead[k]);
			}
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[1].first = "MiddleForehead";
			skipPoints = std::vector<int>{ 2,6 };
			getExtendedPoints(dataPoints, DermaRegions[1].second, skipPoints);

			dataPoints.clear();
			for (int k = 0; k < LeftForeHead.size(); k++)
			{
				dataPoints.push_back(LeftForeHead[k]);
			}
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[2].first = "LeftForeHead";
			skipPoints = std::vector<int>{ 5 };
			getExtendedPoints(dataPoints, DermaRegions[2].second, skipPoints);

			dataPoints.clear();
			for (int k = 0; k < glabella.size(); k++)
			{
				dataPoints.push_back(glabella[k]);
			}
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[3].first = "Glabella";
			getExtendedPoints(dataPoints, DermaRegions[3].second);

			dataPoints.clear();
			for (int k = 0; k < Nose_Derma.size(); k++)
			{
				dataPoints.push_back(Nose_Derma[k]);
			}
			std::reverse(dataPoints.begin(), dataPoints.end());
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[4].first = "Nose";
			getExtendedPoints(dataPoints, DermaRegions[4].second);

			dataPoints.clear();
			for (int k = 0; k < left_cheek_mouth.size(); k++)
			{
				dataPoints.push_back(left_cheek_mouth[k]);
			}
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[5].first = "CheekLeft";
			getExtendedPoints(dataPoints, DermaRegions[5].second);

			dataPoints.clear();
			for (int k = 0; k < left_cheek_inner.size(); k++)
			{
				dataPoints.push_back(left_cheek_inner[k]);
			}
			std::reverse(dataPoints.begin(), dataPoints.end());
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[6].first = "CheekLeftCynus";
			getExtendedPoints(dataPoints, DermaRegions[6].second);

			dataPoints.clear();
			for (int k = 0; k < right_cheek_mouth.size(); k++)
			{
				dataPoints.push_back(right_cheek_mouth[k]);
			}
			std::reverse(dataPoints.begin(), dataPoints.end());
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[7].first = "CheekRight";
			getExtendedPoints(dataPoints, DermaRegions[7].second);

			dataPoints.clear();
			for (int k = 0; k < right_cheek_inner.size(); k++)
			{
				dataPoints.push_back(right_cheek_inner[k]);
			}
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[8].first = "CheekRightCynus";
			getExtendedPoints(dataPoints, DermaRegions[8].second);

			dataPoints.clear();
			for (int k = 0; k < nasolabialLeft.size() - 2; k++)
			{
				dataPoints.push_back(nasolabialLeft[k]);
			}
			dataPoints.push_back(cv::Point(nasolabialLeft[7].x + (int)(0.5*(nasolabialLeft[8].x - nasolabialLeft[7].x)), nasolabialLeft[7].y + (int)(0.5*(nasolabialLeft[8].y - nasolabialLeft[7].y))));
			DermaRegions[9].first = "NasolabialLeft";
			DermaRegions[9].second = dataPoints;

			dataPoints.clear();
			for (int k = 0; k < nasolabialRight.size() - 2; k++)
			{
				dataPoints.push_back(nasolabialRight[k]);
			}
			dataPoints.push_back(cv::Point(nasolabialRight[7].x + (int)(0.5*(nasolabialRight[8].x - nasolabialRight[7].x)), nasolabialRight[7].y + (int)(0.5*(nasolabialRight[8].y - nasolabialRight[7].y))));
			DermaRegions[10].first = "NasolabialRight";
			DermaRegions[10].second = dataPoints;

			dataPoints.clear();
			for (int k = 0; k < chin.size(); k++)
			{
				dataPoints.push_back(Chin_Derma[k]);
			}
			dataPoints.push_back(dataPoints[0]);
			DermaRegions[11].first = "Chin";
			getExtendedPoints(dataPoints, DermaRegions[11].second);
		}
		else if (no_of_regions == 21)
		{
			std::vector<cv::Point> rightforehead, rightmiddleforehead, leftmiddleforehead, leftforehead;
			rightforehead.push_back(forehead[0]);
			rightforehead.push_back(forehead[1]);
			rightforehead.push_back(forehead[2]);
			rightforehead.push_back(forehead[12]);
			rightforehead.push_back(forehead[13]);
			rightforehead.push_back(forehead[14]);

			cv::Point forehead_middle_point;
			forehead_middle_point.y = (int)(0.5*(forehead[11].y + forehead[12].y));
			if (forheadverticalslope != 0)
			{
				forehead_middle_point.x = fldPoints[27].x + (int)((forehead_middle_point.y - fldPoints[27].y) / forheadverticalslope);
			}
			else
			{
				forehead_middle_point.x = fldPoints[27].x;
			}

			rightmiddleforehead.push_back(forehead[2]);
			rightmiddleforehead.push_back(forehead[3]);
			rightmiddleforehead.push_back(forehead_middle_point);
			rightmiddleforehead.push_back(forehead[12]);

			leftmiddleforehead.push_back(forehead[3]);
			leftmiddleforehead.push_back(forehead[4]);
			leftmiddleforehead.push_back(forehead[9]);
			leftmiddleforehead.push_back(forehead_middle_point);


			leftforehead.push_back(forehead[4]);
			leftforehead.push_back(forehead[5]);
			leftforehead.push_back(forehead[6]);
			leftforehead.push_back(forehead[7]);
			leftforehead.push_back(forehead[8]);
			leftforehead.push_back(forehead[9]);

			std::vector<cv::Point> right_nose_derma, left_nose_derma;
			right_nose_derma.push_back(fldPoints[27]);
			right_nose_derma.push_back(noseforredness[0]);
			right_nose_derma.push_back(noseforredness[1]);
			right_nose_derma.push_back(noseforredness[2]);
			right_nose_derma.push_back(noseforredness[3]);
			right_nose_derma.push_back(cv::Point(fldPoints[30].x, (int)(0.5*(fldPoints[33].y + fldPoints[30].y))));

			left_nose_derma.push_back(cv::Point(fldPoints[30].x, (int)(0.5*(fldPoints[33].y + fldPoints[30].y))));
			left_nose_derma.push_back(noseforredness[11]);
			left_nose_derma.push_back(noseforredness[12]);
			left_nose_derma.push_back(noseforredness[13]);
			left_nose_derma.push_back(noseforredness[14]);
			left_nose_derma.push_back(fldPoints[27]);

			std::vector<cv::Point> right_cheek_ue, right_cheek_mouth, right_cheek_jowl, right_cheek_cynus;


			right_cheek_cynus.push_back(cheekRight[2]);
			right_cheek_cynus.push_back(cheekRight[1]);
			for (int k = 1; k <= 3; k++)
			{
				right_cheek_cynus.push_back(noseforredness[k]);
			}
			if (noseforredness[4].y > noseforredness[3].y)
			{
				right_cheek_cynus.push_back(noseforredness[4]);
			}
			right_cheek_cynus.push_back(nasolabialRight[8]);

			//    cv::fillConvexPoly(ROIMask, right_cheek_cynus, cv::Scalar(255, 255, 0), CV_AA);

			double cheek_right_slope = 0;
			if ((cheekRight[3].x - cheekRight[2].x) != 0)
			{
				cheek_right_slope = (cheekRight[3].y - cheekRight[2].y) / (double)(cheekRight[3].x - cheekRight[2].x);
			}
			right_cheek_ue.push_back(nasolabialRight[8]);
			right_cheek_ue.push_back(cheekRight[2]);
			int index = 0;
			double right_cheek_ue_y = nasolabialRight[8].y + cheek_right_slope * (cheekRight[4].x - nasolabialRight[8].x);
			for (int k = 3; k <= 7; k++)
			{
				if (cheekRight[k].y < right_cheek_ue_y) {
					right_cheek_ue.push_back(cheekRight[k]);
					index = k;
				}
			}
			cv::Point intersectionpoint = line_intersection_intercept(cheekRight[index], cheekRight[index + 1], nasolabialRight[8], cheek_right_slope);
			right_cheek_ue.push_back(intersectionpoint);
			//cv::fillConvexPoly(ROIMask, right_cheek_ue, cv::Scalar(255, 255, 0), CV_AA);

			right_cheek_mouth.push_back(right_cheek_ue[right_cheek_ue.size() - 1]);
			double right_cheek_mouth_y = nasolabialRight[6].y + cheek_right_slope * (cheekRight[5].x - nasolabialRight[6].x);
			for (int k = 8; k >= 6; k--)
			{
				right_cheek_mouth.push_back(nasolabialRight[k]);
			}
			index++;
			for (int k = index; k <= 7; k++)
			{
				if (cheekRight[k].y < right_cheek_mouth_y) {
					index = k;
				}
			}
			intersectionpoint = line_intersection_intercept(cheekRight[index], cheekRight[index + 1], nasolabialRight[6], cheek_right_slope);
			right_cheek_mouth.push_back(intersectionpoint);
			//cv::fillConvexPoly(ROIMask, right_cheek_mouth, cv::Scalar(255, 100, 20), CV_AA);


			right_cheek_jowl.push_back(right_cheek_mouth[right_cheek_mouth.size() - 1]);
			index++;
			while (index <= 7) {
				if (cheekRight[index].y > right_cheek_jowl[0].y) {
					right_cheek_jowl.push_back(cheekRight[index]);
				}
				index++;
			}
			right_cheek_jowl.push_back(cheekRight[8]);
			right_cheek_jowl.push_back(cheekRight[9]);
			right_cheek_jowl.push_back(nasolabialRight[5]);
			right_cheek_jowl.push_back(nasolabialRight[6]);

			//cv::fillConvexPoly(ROIMask, right_cheek_jowl, cv::Scalar(255, 0, 0), CV_AA);


			std::vector<cv::Point> left_cheek_ue, left_cheek_mouth, left_cheek_jowl, left_cheek_cynus;


			left_cheek_cynus.push_back(cheekLeft[2]);
			left_cheek_cynus.push_back(cheekLeft[1]);
			for (int k = 13; k >= 10; k--)
			{
				left_cheek_cynus.push_back(noseforredness[k]);
			}
			left_cheek_cynus.push_back(nasolabialLeft[8]);


			double cheek_left_slope = 0;
			if ((cheekLeft[3].x - cheekLeft[2].x) != 0)
			{
				cheek_left_slope = (cheekLeft[3].y - cheekLeft[2].y) / (double)(cheekLeft[3].x - cheekLeft[2].x);
			}

			left_cheek_ue.push_back(nasolabialLeft[8]);
			left_cheek_ue.push_back(cheekLeft[2]);
			double left_cheek_ue_y = nasolabialLeft[8].y + cheek_left_slope * (cheekLeft[4].x - nasolabialLeft[8].x);

			for (int k = 3; k <= 7; k++) {
				if (cheekLeft[k].y < left_cheek_ue_y) {
					left_cheek_ue.push_back(cheekLeft[k]);
					index = k;
				}
			}
			intersectionpoint = line_intersection_intercept(cheekLeft[index], cheekLeft[index + 1], nasolabialLeft[8], cheek_left_slope);
			left_cheek_ue.push_back(intersectionpoint);

			left_cheek_mouth.push_back(left_cheek_ue[left_cheek_ue.size() - 1]);
			double left_cheek_mouth_y = nasolabialLeft[6].y + cheek_left_slope * (cheekLeft[5].x - nasolabialLeft[6].x);
			left_cheek_mouth.push_back(nasolabialLeft[8]);
			left_cheek_mouth.push_back(nasolabialLeft[7]);
			left_cheek_mouth.push_back(nasolabialLeft[6]);
			for (int k = index + 1; k <= 7; k++) {
				if (cheekLeft[k].y < left_cheek_mouth_y) {
					/*left_cheek_mouth[size * 2] = cheekRight[k * 2];
					 left_cheek_mouth[size * 2 + 1] = cheekRight[k * 2 + 1];*/
					index = k;
				}
			}
			intersectionpoint = line_intersection_intercept(cheekLeft[index], cheekLeft[index + 1], nasolabialLeft[6], cheek_left_slope);
			left_cheek_mouth.push_back(intersectionpoint);


			left_cheek_jowl.push_back(left_cheek_mouth[left_cheek_mouth.size() - 1]);
			index++;
			while (index <= 7) {
				if (cheekLeft[index].y > left_cheek_jowl[0].y) {
					left_cheek_jowl.push_back(cheekLeft[index]);
				}
				index++;
			}
			left_cheek_jowl.push_back(cheekLeft[8]);
			left_cheek_jowl.push_back(cheekLeft[9]);
			left_cheek_jowl.push_back(nasolabialLeft[5]);
			left_cheek_jowl.push_back(nasolabialLeft[6]);


			std::vector<cv::Point> nasolabial_right_up, nasolabial_right_down, nasolabial_left_up, nasolabial_left_down;
			for (int k = 0; k <= 2; k++)
			{
				nasolabial_right_up.push_back(nasolabialRight[k]);
			}
			for (int k = 6; k <= 8; k++)
			{
				nasolabial_right_up.push_back(nasolabialRight[k]);
			}

			for (int k = 2; k <= 6; k++)
			{
				nasolabial_right_down.push_back(nasolabialRight[k]);
			}

			for (int k = 0; k <= 2; k++)
			{
				nasolabial_left_up.push_back(nasolabialLeft[k]);
			}
			for (int k = 6; k <= 8; k++)
			{
				nasolabial_left_up.push_back(nasolabialLeft[k]);
			}

			for (int k = 2; k <= 6; k++)
			{
				nasolabial_left_down.push_back(nasolabialLeft[k]);
			}

			cv::Point chin_middle_point = chin[3];
			if (forheadverticalslope != 0)
			{
				chin_middle_point.x = fldPoints[27].x + (int)((chin_middle_point.y - fldPoints[27].y) / forheadverticalslope);
			}
			else
			{
				chin_middle_point.x = fldPoints[27].x;
			}
			std::vector<cv::Point> right_chin_derma, left_chin_derma;

			left_chin_derma.push_back(chin_middle_point);
			for (int k = 4; k <= 5; k++)
			{
				left_chin_derma.push_back(chin[k]);
			}
			left_chin_derma.push_back(nasolabialLeft[4]);

			for (int k = 7; k <= 9; k++)
			{
				left_chin_derma.push_back(chin[k]);
			}

			right_chin_derma.push_back(nasolabialRight[4]);
			for (int k = 1; k <= 2; k++)
			{
				right_chin_derma.push_back(chin[k]);
			}
			right_chin_derma.push_back(chin_middle_point);

			for (int k = 9; k <= 11; k++)
			{
				right_chin_derma.push_back(chin[k]);
			}


			for (int k = 0; k < rightforehead.size(); k++)
			{
				DermaRegions[0].second.push_back(rightforehead[k]);
			}
			DermaRegions[0].first = "RightForehead";

			for (int k = 0; k < rightmiddleforehead.size(); k++)
			{
				DermaRegions[1].second.push_back(rightmiddleforehead[k]);
			}
			DermaRegions[1].first = "RightMiddleForehead";

			for (int k = 0; k < leftmiddleforehead.size(); k++)
			{
				DermaRegions[2].second.push_back(leftmiddleforehead[k]);
			}
			DermaRegions[2].first = "LeftMiddleForehead";

			for (int k = 0; k < leftforehead.size(); k++)
			{
				DermaRegions[3].second.push_back(leftforehead[k]);
			}
			DermaRegions[3].first = "LeftForehead";

			for (int k = 0; k < glabella.size(); k++)
			{
				DermaRegions[4].second.push_back(glabella[k]);
			}
			DermaRegions[4].first = "Glabella";

			for (int k = 0; k < right_nose_derma.size(); k++)
			{
				DermaRegions[5].second.push_back(right_nose_derma[k]);
			}
			DermaRegions[5].first = "RightNoseDerma";

			for (int k = 0; k < left_nose_derma.size(); k++)
			{
				DermaRegions[6].second.push_back(left_nose_derma[k]);
			}
			DermaRegions[6].first = "LeftNoseDerma";

			for (int k = 0; k < right_cheek_ue.size(); k++)
			{
				DermaRegions[7].second.push_back(right_cheek_ue[k]);
			}
			DermaRegions[7].first = "RightCheekUE";

			for (int k = 0; k < right_cheek_mouth.size(); k++)
			{
				DermaRegions[8].second.push_back(right_cheek_mouth[k]);
			}
			DermaRegions[8].first = "RightCheekMouth";

			for (int k = 0; k < right_cheek_jowl.size(); k++)
			{
				DermaRegions[9].second.push_back(right_cheek_jowl[k]);
			}
			DermaRegions[9].first = "RightCheekJowl";

			for (int k = 0; k < right_cheek_cynus.size(); k++)
			{
				DermaRegions[10].second.push_back(right_cheek_cynus[k]);
			}
			DermaRegions[10].first = "RightCheekCynus";

			for (int k = 0; k < left_cheek_ue.size(); k++)
			{
				DermaRegions[11].second.push_back(left_cheek_ue[k]);
			}
			DermaRegions[11].first = "LeftCheekUE";

			for (int k = 0; k < left_cheek_mouth.size(); k++)
			{
				DermaRegions[12].second.push_back(left_cheek_mouth[k]);
			}
			DermaRegions[12].first = "LeftCheekMouth";

			for (int k = 0; k < left_cheek_jowl.size(); k++)
			{
				DermaRegions[13].second.push_back(left_cheek_jowl[k]);
			}
			DermaRegions[13].first = "LeftCheekJowl";

			for (int k = 0; k < left_cheek_cynus.size(); k++)
			{
				DermaRegions[14].second.push_back(left_cheek_cynus[k]);
			}
			DermaRegions[14].first = "LeftCheekCynus";

			for (int k = 0; k < nasolabial_right_up.size(); k++)
			{
				DermaRegions[15].second.push_back(nasolabial_right_up[k]);
			}
			DermaRegions[15].first = "NasolabialRightUP";

			for (int k = 0; k < nasolabial_right_down.size(); k++)
			{
				DermaRegions[16].second.push_back(nasolabial_right_down[k]);
			}
			DermaRegions[16].first = "NasolabialRightDown";

			for (int k = 0; k < nasolabial_left_up.size(); k++)
			{
				DermaRegions[17].second.push_back(nasolabial_left_up[k]);
			}
			DermaRegions[17].first = "NasolabialLeftUP";

			for (int k = 0; k < nasolabial_left_down.size(); k++)
			{
				DermaRegions[18].second.push_back(nasolabial_left_down[k]);
			}
			DermaRegions[18].first = "NasolabialLeftDown";

			for (int k = 0; k < right_chin_derma.size(); k++)
			{
				DermaRegions[19].second.push_back(right_chin_derma[k]);
			}
			DermaRegions[19].first = "RightChinDerma";
			for (int k = 0; k < left_chin_derma.size(); k++)
			{
				DermaRegions[20].second.push_back(left_chin_derma[k]);
			}
			DermaRegions[20].first = "LeftChinDerma";
		}
	}
	catch (const std::exception&ex)
	{
		std::cout << ":Exception due to" << ex.what();
	}
	return DermaRegions;
}

cv::Point Skincare::FacialRegions_Live::line_intersection_intercept(cv::Point A, cv::Point B, cv::Point C, double m2)
{
	cv::Point intersect_point;
	double m1 = 0;
	if ((B.x - A.x) != 0) {
		m1 = (B.y - A.y) / (double)(B.x - A.x);
	}
	double c1 = A.y - m1*  A.x, c2 = C.y - m2*  C.x;
	double determinant = m1 - m2;
	if (determinant != 0)
	{
		if (m1 != 0)
		{
			intersect_point.x = (int)((c2 - c1) / (m1 - m2));
		}
		else
		{
			intersect_point.x = B.x;
		}
	}
	else
	{
		intersect_point.x = (int)(c2 - c1);
	}
	intersect_point.y = (int)(m2* intersect_point.x + c2);
	return intersect_point;
}


void Skincare::FacialRegions_Live::getExtendedPoints(std::vector<cv::Point>& dataPoints, std::vector<cv::Point>& extendedPoints, std::vector<int> skipPoints, bool isClockwise) {
	int n = (int)dataPoints.size() - 1;
	for (int i = 0; i < n; i++) {
		extendedPoints.push_back(dataPoints[i]);
		bool exists = std::find(skipPoints.begin(), skipPoints.end(), i) != skipPoints.end();
		if (exists)
			continue;
		if ((dataPoints[i].x < dataPoints[i + 1].x && dataPoints[i].y > dataPoints[i + 1].y) || (dataPoints[i].x > dataPoints[i + 1].x && dataPoints[i].y < dataPoints[i + 1].y)) {
			if (isClockwise)
			{
				int mx = dataPoints[i].x + (int)((dataPoints[i + 1].x - dataPoints[i].x) * 0.25);
				int my = dataPoints[i].y + (int)((dataPoints[i + 1].y - dataPoints[i].y) * 0.35);
				extendedPoints.push_back(cv::Point(mx, my));
				mx = dataPoints[i].x + (int)((dataPoints[i + 1].x - dataPoints[i].x) * 0.70);
				my = dataPoints[i].y + (int)((dataPoints[i + 1].y - dataPoints[i].y) * 0.80);
				extendedPoints.push_back(cv::Point(mx, my));
			}
			else
			{
				int mx = dataPoints[i].x + (int)((dataPoints[i + 1].x - dataPoints[i].x) * 0.35);
				int my = dataPoints[i].y + (int)((dataPoints[i + 1].y - dataPoints[i].y) * 0.25);
				extendedPoints.push_back(cv::Point(mx, my));
				mx = dataPoints[i].x + (int)((dataPoints[i + 1].x - dataPoints[i].x) * 0.80);
				my = dataPoints[i].y + (int)((dataPoints[i + 1].y - dataPoints[i].y) * 0.70);
				extendedPoints.push_back(cv::Point(mx, my));
			}
		}
		else {
			if (isClockwise)
			{
				int mx = dataPoints[i].x + (int)((dataPoints[i + 1].x - dataPoints[i].x) * 0.35);
				int my = dataPoints[i].y + (int)((dataPoints[i + 1].y - dataPoints[i].y) * 0.25);
				extendedPoints.push_back(cv::Point(mx, my));
				mx = dataPoints[i].x + (int)((dataPoints[i + 1].x - dataPoints[i].x) * 0.80);
				my = dataPoints[i].y + (int)((dataPoints[i + 1].y - dataPoints[i].y) * 0.70);
				extendedPoints.push_back(cv::Point(mx, my));
			}
			else
			{
				int mx = dataPoints[i].x + (int)((dataPoints[i + 1].x - dataPoints[i].x) * 0.25);
				int my = dataPoints[i].y + (int)((dataPoints[i + 1].y - dataPoints[i].y) * 0.35);
				extendedPoints.push_back(cv::Point(mx, my));
				mx = dataPoints[i].x + (int)((dataPoints[i + 1].x - dataPoints[i].x) * 0.70);
				my = dataPoints[i].y + (int)((dataPoints[i + 1].y - dataPoints[i].y) * 0.80);
				extendedPoints.push_back(cv::Point(mx, my));
			}
		}
	}
};



