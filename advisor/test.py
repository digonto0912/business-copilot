from agents.critic_agent import CriticAgent


# ============================================================
# CRITIC
# ============================================================

critic = CriticAgent()


# ============================================================
# ADVISOR STRATEGY OUTPUT
# ============================================================

advisor_strategy = {
    "title": "Strategic Pivot for Home Decor Dropshipping in Bangladesh",
    "business_summary": {
        "business_model": "Dropshipping",
        "product_category": "Home decor",
        "price_range_bdt": None,
        "target_customer": "Bangladeshi consumers interested in home decor",
        "market": "Bangladesh",
        "fulfillment": "Supplier handles shipping",
        "constraints": [
            "Zero budget",
            "No original content",
            "Low conversion rate",
            "Lack of social proof",
        ],
    },
    "real_problem": {
        "surface_problem": "Struggling to convert views into inquiries on Facebook.",
        "actual_problem": "Operating as a commodity reseller with no competitive advantage, effectively performing free marketing for the supplier.",
        "supporting_context": [
            "Customers compare sellers, not just products.",
            "Using identical supplier content leads customers to the supplier's page with higher social proof.",
            "A page with 0 followers and stolen content is perceived as a scam.",
        ],
    },
    "strategy_reasoning": {
        "core_constraint": "Lack of trust and competitive differentiation.",
        "approach": "Shift from commodity reselling to a value-added 'Home Decor Consultant' model focused on curation and personal trust.",
        "channels": [
            "Facebook Page",
            "Local interior design and home decor Facebook groups",
        ],
    },
    "prioritized_action_plan": [
        {
            "priority": 1,
            "title": "Community Engagement",
            "action": "Build personal trust by positioning as a consultant in local community groups.",
            "execution": [
                "Join Dhaka-based interior design and home decor Facebook groups.",
                "Answer questions and provide decor advice.",
                "Mention access to specific pieces only after establishing rapport.",
            ],
        },
    ],
    "assumptions_risks_tradeoffs": {
        "assumptions": [
            "The business owner can negotiate with the supplier to send samples to influencers."
        ],
        "risks": [
            {
                "risk": "Continuing to use the supplier's exact content.",
                "impact": "Conversion rate will remain near zero due to lack of trust.",
                "mitigation": None,
            }
        ],
        "tradeoffs": [
            "This strategy requires significantly more time and effort than simply reposting videos."
        ],
    },
}


# ============================================================
# VERIFIED CONTEXT
# ============================================================

verified_context = [
    {
        "qa_pairs": [],
        "unprompted_context": [
            {
                "raw_text": "i want first 100 customers in my business",
                "topic_hint": "business goal",
            }
        ],
    },
    {
        "qa_pairs": [],
        "unprompted_context": [
            {
                "raw_text": "i want first 100 customers in my business",
                "topic_hint": "business goal",
            },
            {
                "raw_text": "i am selling home decore showpices, this is a b2c business and my ideal customer is bangladeshi womans, i have 0 budget for this business and i have no email followe or network",
                "topic_hint": "business details and constraints",
            },
        ],
    },
    {
        "qa_pairs": [],
        "unprompted_context": [
            {
                "raw_text": "i am selling home decore showpices, this is a b2c business and my ideal customer is bangladeshi womans, i have 0 budget for this business and i have no email followe or network",
                "topic_hint": "Business model, target audience, and budget constraints",
            },
            {
                "raw_text": "i am showcasing my product in facebook page, and the logistics and everything done by supplyer, my task is just selling the product rest of others work will done by him, my showpieces price point is 2000-4000, you can buy them from local bazar too but problem is if you live in dhaka then otherwise local bazar will charge you 3000-5000 and most of the items are not abailable in others place but dhaka because regularly people dont buy this items so market not show them regularly",
                "topic_hint": "Sales channel, logistics, and product pricing/availability",
            },
        ],
    },
    {
        "qa_pairs": [],
        "unprompted_context": [
            {
                "raw_text": "i am showcasing my product in facebook page, and the logistics and everything done by supplyer, my task is just selling the product rest of others work will done by him, my showpieces price point is 2000-4000, you can buy them from local bazar too but problem is if you live in dhaka then otherwise local bazar will charge you 3000-5000 and most of the items are not abailable in others place but dhaka because regularly people dont buy this items so market not show them regularly",
                "topic_hint": "business model and product pricing",
            },
            {
                "raw_text": "no i dont have physical access to the product, the supplyer already running a Ggly succesful facebook page where he is uploading his contents and i am taking same successful videos from here and changing audios, and if i share them in groups then i get some view but not reaction or message and also after uploading 40+ videos in last month i achive total 25k views but maybe only 5 reaction inside the group post not in my page and not message at all. and i contact with supplyer he tell me to use his content and his contents are mostly directly showing product in his hand with a bgm, and people are reacting and comment and buying from him, which is a big signal and he has 50k+ followers and top video has 2-5k+ reaction maybe.",
                "topic_hint": "operational process and performance metrics",
            },
        ],
    },
    {
        "qa_pairs": [],
        "unprompted_context": [
            {
                "raw_text": "no i dont have physical access to the product, the supplyer already running a Ggly succesful facebook page where he is uploading his contents and i am taking same successful videos from here and changing audios, and if i share them in groups then i get some view but not reaction or message and also after uploading 40+ videos in last month i achive total 25k views but maybe only 5 reaction inside the group post not in my page and not message at all.",
                "topic_hint": "Business operations and current performance",
            },
            {
                "raw_text": "i contact with supplyer he tell me to use his content and his contents are mostly directly showing product in his hand with a bgm, and people are reacting and comment and buying from him, which is a big signal and he has 50k+ followers and top video has 2-5k+ reaction maybe.",
                "topic_hint": "Supplier relationship and social proof",
            },
            {
                "raw_text": "facebook is not build with 10 peoples network that people can compare with me with my supplyer, facebook network is very big. and that why in ocean if 2 bot try to catch fish that not mean all the fish will go to the big bot because it is big, no ocean has that much fish which can fullfil 1k bots without lossing anything.",
                "topic_hint": "Market competition hypothesis",
            },
            {
                "raw_text": "people are also in facebook not buy i one go, they like to watch same product before buy 7-14 time then they take decision of pucesing or messaging and then compare price end of the day order. so, that not mean when they will see his post they will directly buy from him that can also be happen that the 7 or 14 number video is mine.",
                "topic_hint": "Customer buying behavior",
            },
        ],
    },
]


# ============================================================
# TARGET ACTION
# ============================================================

target_action = advisor_strategy["prioritized_action_plan"][0]


# ============================================================
# CRITIQUE
# ============================================================

critique = critic.critique_action(
    action_item=target_action,
    verified_context=verified_context,
    advisor_strategy={
        key: value
        for key, value in advisor_strategy.items()
        if key != "prioritized_action_plan"
    },
)


# ============================================================
# OUTPUT
# ============================================================

print(
    critique.model_dump_json(
        indent=2,
    )
)
