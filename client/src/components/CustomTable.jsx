import React from "react";
import PropTypes from "prop-types";
import {
	Table,
	TableBody,
	TableCell,
	TableContainer,
	TableFooter,
	TableHead,
	TableRow,
	Paper,
	TablePagination,
	Box,
	IconButton,
	useTheme,
} from "@mui/material";
import FirstPageIcon from "@mui/icons-material/FirstPage";
import LastPageIcon from "@mui/icons-material/LastPage";
import KeyboardArrowLeft from "@mui/icons-material/KeyboardArrowLeft";
import KeyboardArrowRight from "@mui/icons-material/KeyboardArrowRight";

function TablePaginationActions(props) {
	const theme = useTheme();
	const { count, page, rowsPerPage, onPageChange } = props;

	const handleFirstPageButtonClick = (event) => {
		onPageChange(event, 0);
	};

	const handleBackButtonClick = (event) => {
		onPageChange(event, page - 1);
	};

	const handleNextButtonClick = (event) => {
		onPageChange(event, page + 1);
	};

	const handleLastPageButtonClick = (event) => {
		onPageChange(event, Math.max(0, Math.ceil(count / rowsPerPage) - 1));
	};

	return (
		<Box sx={{ flexShrink: 0, ml: 2.5 }}>
			<IconButton
				onClick={handleFirstPageButtonClick}
				disabled={page === 0}
				aria-label="first page"
			>
				{theme.direction === "rtl" ? <LastPageIcon /> : <FirstPageIcon />}
			</IconButton>
			<IconButton
				onClick={handleBackButtonClick}
				disabled={page === 0}
				aria-label="previous page"
			>
				{theme.direction === "rtl" ? (
					<KeyboardArrowRight />
				) : (
					<KeyboardArrowLeft />
				)}
			</IconButton>
			<IconButton
				onClick={handleNextButtonClick}
				disabled={page >= Math.ceil(count / rowsPerPage) - 1}
				aria-label="next page"
			>
				{theme.direction === "rtl" ? (
					<KeyboardArrowLeft />
				) : (
					<KeyboardArrowRight />
				)}
			</IconButton>
			<IconButton
				onClick={handleLastPageButtonClick}
				disabled={page >= Math.ceil(count / rowsPerPage) - 1}
				aria-label="last page"
			>
				{theme.direction === "rtl" ? <FirstPageIcon /> : <LastPageIcon />}
			</IconButton>
		</Box>
	);
}

TablePaginationActions.propTypes = {
	count: PropTypes.number.isRequired,
	onPageChange: PropTypes.func.isRequired,
	page: PropTypes.number.isRequired,
	rowsPerPage: PropTypes.number.isRequired,
};

const headers = [
	{ metric: "NER", value: "Value for NER" },
	{ metric: "Category", value: "Value for Category" },
	{ metric: "Intent", value: "Value for Intent" },
];

const intentLabels = {
	0: "cancel_order",
	1: "change_order",
	2: "change_shipping_address",
	3: "check_cancellation_fee",
	4: "check_invoice",
	5: "check_payment_methods",
	6: "check_refund_policy",
	7: "complaint",
	8: "contact_customer_service",
	9: "contact_human_agent",
	10: "create_account",
	11: "delete_account",
	12: "delivery_options",
	13: "delivery_period",
	14: "edit_account",
	15: "get_invoice",
	16: "get_refund",
	17: "newsletter_subscription",
	18: "payment_issue",
	19: "place_order",
	20: "recover_password",
	21: "registration_problems",
	22: "review",
	23: "set_up_shipping_address",
	24: "switch_account",
	25: "track_order",
	26: "track_refund",
};

const categoryLabels = {
	0: "ORDER",
	1: "SHIPPING",
	2: "CANCEL",
	3: "INVOICE",
	4: "PAYMENT",
	5: "REFUND",
	6: "FEEDBACK",
	7: "CONTACT",
	8: "ACCOUNT",
	9: "DELIVERY",
	10: "SUBSCRIPTION",
};

const CustomTable = ({ categoryResponse, intentResponse, nerResponse }) => {
	console.log("table", categoryResponse, intentResponse, nerResponse);
	const [page, setPage] = React.useState(0);
	const [rowsPerPage, setRowsPerPage] = React.useState(3);

	const handleChangePage = (event, newPage) => {
		setPage(newPage);
	};

	const handleChangeRowsPerPage = (event) => {
		setRowsPerPage(parseInt(event.target.value, 10));
		setPage(0);
	};

	// Function to get top 3 data points and convert to percentages
	const getTop3Data = (data, labels) => {
		if (!Array.isArray(data)) return [];
		return data
			.map((value, index) => ({ value, label: labels[index] }))
			.sort((a, b) => b.value - a.value)
			.slice(0, 3)
			.map((item, index) => ({
				metric: item.label,
				value: `${(item.value * 100).toFixed(2)}%`,
			}));
	};

	const getTop3NerData = (data) => {
		if (!Array.isArray(data)) return [];
		return data
			.slice()
			.sort((a, b) => b - a)
			.slice(0, 3)
			.map((value, index) => ({
				metric: `NER ${index + 1}`,
				value: `${(value * 100).toFixed(2)}%`,
			}));
	};

	const top3CategoryResponse = getTop3Data(categoryResponse, categoryLabels);
	const top3IntentResponse = getTop3Data(intentResponse, intentLabels);
	const top3NerResponse = getTop3NerData(nerResponse);

	// Determine the current header based on the page
	const currentHeader = headers[page % headers.length];

	// Combine all top 3 data points
	const combinedData = [
		...top3NerResponse,
		...top3CategoryResponse,
		...top3IntentResponse,
	];

	return (
		<TableContainer component={Paper}>
			<Table sx={{ minWidth: 500 }} aria-label="custom pagination table">
				<TableHead>
					<TableRow>
						<TableCell>{currentHeader.metric}</TableCell>
						<TableCell align="right">{currentHeader.value}</TableCell>
					</TableRow>
				</TableHead>
				<TableBody>
					{(rowsPerPage > 0
						? combinedData.slice(
								page * rowsPerPage,
								page * rowsPerPage + rowsPerPage
						  )
						: combinedData
					).map((row, index) => (
						<TableRow key={index}>
							<TableCell component="th" scope="row">
								{row.metric}
							</TableCell>
							<TableCell align="right">{row.value}</TableCell>
						</TableRow>
					))}
				</TableBody>
				<TableFooter>
					<TableRow>
						<TablePagination
							rowsPerPageOptions={[3, 5, 10]}
							colSpan={3}
							count={combinedData.length}
							rowsPerPage={rowsPerPage}
							page={page}
							SelectProps={{
								inputProps: {
									"aria-label": "rows per page",
								},
								native: true,
							}}
							onPageChange={handleChangePage}
							onRowsPerPageChange={handleChangeRowsPerPage}
							ActionsComponent={TablePaginationActions}
						/>
					</TableRow>
				</TableFooter>
			</Table>
		</TableContainer>
	);
};

CustomTable.propTypes = {
	categoryResponse: PropTypes.arrayOf(PropTypes.number).isRequired,
	intentResponse: PropTypes.arrayOf(PropTypes.number).isRequired,
	nerResponse: PropTypes.arrayOf(PropTypes.number).isRequired,
};

// Default data
CustomTable.defaultProps = {
	categoryResponse: [
		0.92313, 0.21511, 0.08318, 0.14071, 0.04769, 0.03593, 0.0152, 0.01186,
		0.0809, 0.05934, 0.07558,
	],
	intentResponse: [
		0.92313, 0.21511, 0.08318, 0.14071, 0.04769, 0.03593, 0.0152, 0.01186,
		0.0809, 0.05934, 0.07558,
	],
	nerResponse: [
		0.92313, 0.21511, 0.08318, 0.14071, 0.04769, 0.03593, 0.0152, 0.01186,
		0.0809, 0.05934, 0.07558,
	],
};

export default CustomTable;
