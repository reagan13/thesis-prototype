import Page from "./Page";

const names = ["Baseline", "Proposed Solution"];

const Distilbert = () => {
	return (
		<div className="flex-grow grid grid-cols-1 md:grid-cols-2 gap-10 p-6">
			{names.map((name, index) => (
				<Page
					key={index}
					name={name}
					response={`Response for ${name}`}
					categorization={`Category of ${name}`}
					analysis={`Analysis of ${name}`}
				/>
			))}
		</div>
	);
};

export default Distilbert;
